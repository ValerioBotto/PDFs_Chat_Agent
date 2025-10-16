"""
Cross-document context helper utilities (MCP-only).

Goal: when the active document doesn't provide enough context, discover
and fetch relevant chunks from other documents in Neo4j (same or related topics),
using MCP tools to execute Cypher queries. No direct GraphDB client usage here.

This module is intentionally lightweight and decoupled from the agent graph.
It exposes pure helpers that can be called from agent nodes or tools.

Key functions:
- decide_insufficient: heuristics to decide whether primary search is insufficient.
- get_active_topics_via_mcp: fetch topics for the active document via MCP.
- find_related_docs_via_mcp: discover related documents using MCP read-cypher.
- retrieve_chunks_for_docs: vector search per candidate document, using MCP.
- build_augmented_context: merge/format primary and cross-doc results.
- crossdoc_augmented_search: orchestration end-to-end.

Notes:
- MCP tools are required. If missing, helpers return empty results instead of falling back.
- We avoid importing from agent.py to prevent circular imports.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.pdf_utils.indexer import Indexer

logger = logging.getLogger(__name__)


# ------------------------------
# Heuristics: insufficient context
# ------------------------------

def decide_insufficient(
    primary_results: Optional[Sequence[Dict[str, Any]]],
    *,
    min_results: int = 3,
    min_avg_score: float = 0.20,
    allow_if_contains_answer_like: bool = True,
) -> Tuple[bool, str]:
    """
    Decide if the initial vector-search results from the active document are insufficient.

    Inputs:
    - primary_results: list of dicts as returned by your vector search layer
      [{'node_content', 'score', 'chunk_id', 'section', 'filename'}, ...]

    Returns: (insufficient: bool, reason: str)
    """
    try:
        if not primary_results:
            return True, "no_results_from_active_document"
        n = len(primary_results)
        if n < min_results:
            return True, f"too_few_results({n}<{min_results})"
        # safety for missing scores
        scores = [float(r.get("score", 0.0)) for r in primary_results if isinstance(r, dict)]
        if not scores:
            return True, "no_scores_in_results"
        avg_score = sum(scores) / len(scores)
        if avg_score < min_avg_score:
            return True, f"low_avg_score({avg_score:.3f}<{min_avg_score})"

        if allow_if_contains_answer_like:
            # quick textual heuristic: presence of explicit answer-like sentences
            text_blob = "\n".join(str(r.get("node_content", "")) for r in primary_results if isinstance(r, dict))
            lowered = text_blob.lower()
            # If content looks like it includes definitions/explanations, consider it sufficient
            hints = ("in conclusione", "in summary", "definizione", "definition", "conclude")
            if any(h in lowered for h in hints):
                return False, "answer_like_content_present"
        return False, "sufficient_context"
    except Exception as e:
        logger.warning(f"decide_insufficient failed, defaulting to cross-doc: {e}")
        return True, "heuristic_error"


# ------------------------------
# MCP helpers
# ------------------------------

def _select_mcp_tool(tools: Optional[List[Any]], candidate_names: List[str]) -> Optional[Any]:
    """Pick the first MCP tool whose .name matches one of candidate_names or endswith('-<candidate>')."""
    if not tools:
        return None
    names = set(candidate_names)
    try:
        for t in tools:
            try:
                n = getattr(t, "name", None) or getattr(t, "__name__", None)
            except Exception:
                n = None
            if not n:
                continue
            if n in names:
                return t
            # support namespaced like "local-read_neo4j_cypher"
            for cand in candidate_names:
                if n.endswith("-" + cand) or n.endswith("_" + cand):
                    return t
    except Exception as e:
        logger.debug(f"_select_mcp_tool error: {e}")
    return None


async def _mcp_read_cypher(
    tools: Optional[List[Any]],
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout: Optional[float] = None,
    candidate_names: Optional[List[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Invoke MCP read-cypher tool and parse result into list[dict]. Returns None on failure.
    Expected tool input: {"query": <cypher>, "params": <dict>}.
    """
    candidate_names = candidate_names or ["read_neo4j_cypher", "neo4j_read_cypher"]
    tool = _select_mcp_tool(tools, candidate_names)
    if not tool:
        return None

    payload = {"query": cypher, "params": params or {}}
    try:
        # Try async path first
        if hasattr(tool, "ainvoke"):
            coro = tool.ainvoke(payload)
            if timeout and timeout > 0:
                raw = await asyncio.wait_for(coro, timeout=timeout)
            else:
                raw = await coro
        else:
            # Sync path, but avoid blocking the loop if called from async
            def _call_sync():
                if hasattr(tool, "invoke"):
                    return tool.invoke(payload)
                return None

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    raw = await asyncio.to_thread(_call_sync)
                else:
                    raw = _call_sync()
            except RuntimeError:
                # No running loop
                raw = _call_sync()

        # Parse: many MCP tools return list[dict], or JSON string
        if raw is None:
            return None
        if isinstance(raw, list):
            # already in structured form
            return [dict(r) if not isinstance(r, dict) else r for r in raw]
        if isinstance(raw, dict):
            # sometimes tools return {"rows": [...]} or similar
            if "rows" in raw and isinstance(raw["rows"], list):
                return [dict(r) if not isinstance(r, dict) else r for r in raw["rows"]]
            # fallback: single-row dict
            return [raw]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [dict(r) if not isinstance(r, dict) else r for r in parsed]
                if isinstance(parsed, dict):
                    if "rows" in parsed and isinstance(parsed["rows"], list):
                        return [dict(r) if not isinstance(r, dict) else r for r in parsed["rows"]]
                    return [parsed]
            except Exception:
                # last resort: not JSON
                logger.debug("_mcp_read_cypher: non-JSON string result; returning None")
                return None
        # Unknown type
        return None
    except Exception as e:
        logger.warning(f"_mcp_read_cypher failed: {e}")
        return None


# ------------------------------
# Related document discovery (MCP-only)
# ------------------------------

async def get_active_topics_via_mcp(
    active_filename: str,
    mcp_tools: Optional[List[Any]],
    *,
    timeout: Optional[float] = None,
) -> Optional[List[str]]:
    """Fetch topics attached to the active document via MCP; returns lowercased names."""
    cypher = (
        "MATCH (d:Document {filename: $filename})-[:HAS_TOPIC]->(t:Topic) "
        "RETURN collect(distinct toLower(t.name)) AS topics"
    )
    rows = await _mcp_read_cypher(mcp_tools, cypher, {"filename": active_filename}, timeout=timeout)
    if not rows:
        return None
    try:
        row = rows[0]
        topics = row.get("topics") if isinstance(row, dict) else None
        if isinstance(topics, list):
            return [str(x).lower() for x in topics if x]
    except Exception:
        pass
    return None

async def find_related_docs_via_mcp(
    active_filename: str,
    mcp_tools: Optional[List[Any]],
    *,
    limit: int = 3,
    timeout: Optional[float] = None,
) -> List[Tuple[str, int]]:
    """
    Use MCP read-cypher to find other documents that share topics with the active one.
    Returns list of (filename, overlap_count) sorted by overlap desc.
    """
    # First, try overlap-based discovery
    cypher = (
        "MATCH (d:Document {filename: $filename})-[:HAS_TOPIC]->(t:Topic) "
        "MATCH (other:Document)-[:HAS_TOPIC]->(t) "
        "WHERE other.filename <> $filename "
        "WITH other, count(DISTINCT t) AS overlap "
        "ORDER BY overlap DESC "
        "RETURN other.filename AS filename, overlap "
        "LIMIT $limit"
    )
    rows = await _mcp_read_cypher(
        mcp_tools,
        cypher,
        {"filename": active_filename, "limit": limit},
        timeout=timeout,
    )
    out: List[Tuple[str, int]] = []
    if rows:
        try:
            for r in rows:
                fn = r.get("filename") if isinstance(r, dict) else None
                ov = r.get("overlap") if isinstance(r, dict) else None
                if fn and ov is not None:
                    out.append((str(fn), int(ov)))
        except Exception:
            pass
    return out


    


# ------------------------------
# Chunk retrieval per document (MCP-only)
# ------------------------------

async def retrieve_chunks_for_docs(
    question: str,
    candidate_filenames: Sequence[str],
    *,
    indexer: Indexer,
    mcp_tools: Optional[List[Any]],
    k_per_doc: int = 4,
    index_name: str = "chunk_embeddings_index",
) -> List[Dict[str, Any]]:
    """
    For each candidate document, run a vector search scoped to that document using MCP.
    Uses CALL db.index.vector.queryNodes via the MCP read-cypher tool.
    """
    if not candidate_filenames or not mcp_tools:
        return []
    q_emb = indexer.generate_embeddings(question)
    all_res: List[Dict[str, Any]] = []
    for fn in candidate_filenames:
        cypher = (
            "CALL db.index.vector.queryNodes($index_name, $k, $query_embedding) "
            "YIELD node, score "
            "WITH node, score "
            "MATCH (d:Document {filename: $filename})-[:HAS_CHUNK]->(node) "
            "RETURN node.content AS node_content, score, node.chunk_id AS chunk_id, "
            "node.section AS section, d.filename AS filename"
        )
        params = {
            "index_name": index_name,
            "k": int(k_per_doc),
            "query_embedding": q_emb,
            "filename": fn,
        }
        try:
            rows = await _mcp_read_cypher(mcp_tools, cypher, params)
            if rows:
                for r in rows:
                    if isinstance(r, dict):
                        all_res.append({
                            "node_content": r.get("node_content"),
                            "score": float(r.get("score", 0.0)),
                            "chunk_id": r.get("chunk_id"),
                            "section": r.get("section"),
                            "filename": r.get("filename"),
                        })
        except Exception as e:
            logger.warning(f"MCP vector retrieval for '{fn}' failed: {e}")
    # sort by score desc and deduplicate by (filename, chunk_id)
    seen: set[Tuple[str, str]] = set()
    dedup: List[Dict[str, Any]] = []
    for r in sorted(all_res, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        key = (str(r.get("filename")), str(r.get("chunk_id")))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


# ------------------------------
# Merge/format utilities
# ------------------------------

def build_augmented_context(
    primary_results: Optional[Sequence[Dict[str, Any]]],
    crossdoc_results: Optional[Sequence[Dict[str, Any]]],
    *,
    max_total: int = 12,
) -> Dict[str, Any]:
    """
    Merge primary and cross-doc results, preferring higher scores while keeping source labels.
    Returns a dict with unified 'results' and a 'text' field ready for LLM consumption.
    """
    primary = list(primary_results or [])
    cross = list(crossdoc_results or [])
    combined = primary + cross
    # sort by score desc, keep top N unique
    seen: set[Tuple[str, str]] = set()
    uniq: List[Dict[str, Any]] = []
    for r in sorted(combined, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        key = (str(r.get("filename")), str(r.get("chunk_id")))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
        if len(uniq) >= max_total:
            break

    # Build a textual context block
    lines: List[str] = []
    for i, r in enumerate(uniq, start=1):
        fn = r.get("filename", "unknown")
        sec = r.get("section", "")
        sc = r.get("score", 0.0)
        content = r.get("node_content", "")
        header = f"[#{i}] file={fn} section={sec} score={sc:.3f}"
        lines.append(header)
        lines.append(str(content))
        lines.append("")

    return {"results": uniq, "text": "\n".join(lines).strip()}


# ------------------------------
# Orchestration
# ------------------------------

async def crossdoc_augmented_search(
    question: str,
    active_filename: str,
    *,
    indexer: Indexer,
    mcp_tools: Optional[List[Any]] = None,
    candidate_docs_limit: int = 3,
    k_per_doc: int = 4,
    mcp_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Discover related docs (via MCP) and retrieve top-k chunks per doc using MCP.
    Returns the same structure as build_augmented_context for convenience:
    {"results": [...], "text": "..."}.
    """
    if not question or not active_filename:
        return {"results": [], "text": ""}

    # Step 1: discover candidate docs (MCP required)
    candidates: List[str] = []
    try:
        if mcp_tools:
            pairs = await find_related_docs_via_mcp(active_filename, mcp_tools, limit=candidate_docs_limit, timeout=mcp_timeout)
            candidates = [fn for fn, _ in pairs if fn]
    except Exception as e:
        logger.warning(f"candidate discovery failed, fallback to empty: {e}")
        candidates = []

    if not candidates:
        return {"results": [], "text": ""}

    # Step 2: retrieve chunks per candidate doc (MCP required)
    try:
        cross = await retrieve_chunks_for_docs(
            question,
            candidates,
            indexer=indexer,
            mcp_tools=mcp_tools,
            k_per_doc=k_per_doc,
        )
    except Exception as e:
        logger.warning(f"retrieve_chunks_for_docs failed: {e}")
        cross = []

    return {"results": cross, "text": "\n".join([str(r.get("node_content", "")) for r in cross])}
