# -*- coding: utf-8 -*-
"""
RAG 检索模块
- 使用 bge-m3 对问题向量化
- 在 ChromaDB 中检索 Top-K 相关文档
- 返回格式化的检索结果字符串，注入 System Prompt
"""
from __future__ import annotations
import logging
import os
import threading
from typing import Optional
import pdfplumber
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, RAG_TOP_K, CHROMA_DIR, RAG_DISTANCE_THRESHOLD, PDF_UPLOAD_DIRS

log = logging.getLogger(__name__)

# ─── 路径安全检查 ─────────────────────────────────────────────────────────────

def _is_safe_path(file_path: str) -> bool:
    """
    检查文件路径是否在允许的白名单目录内，防止路径遍历攻击。
    """
    try:
        abs_path = os.path.abspath(file_path)
        # 检查文件是否在允许的目录内
        for allowed_dir in PDF_UPLOAD_DIRS:
            allowed_abs = os.path.abspath(allowed_dir)
            # 确保路径在白名单目录下
            if abs_path.startswith(allowed_abs + os.sep) or abs_path == allowed_abs:
                return True
        # 对于 Gradio 上传的临时文件，也允许（通常在 /tmp 或 /tmp/gradio）
        # 检查是否是临时目录下的文件
        if "/tmp/" in abs_path.replace("\\", "/") or "/var/folders/" in abs_path.replace("\\", "/"):
            return True
        return False
    except Exception as e:
        log.warning(f"路径安全检查失败: {e}")
        return False

# 全局单例（应用启动时初始化，避免重复加载）
_embed_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[object] = None
_collection: Optional[object] = None

# 线程锁，保护单例初始化
_init_lock = threading.Lock()


def _get_embed_model() -> SentenceTransformer:
    """获取向量模型单例，使用锁保护初始化"""
    global _embed_model
    if _embed_model is None:
        with _init_lock:
            # 双重检查
            if _embed_model is None:
                log.info(f"加载向量模型: {EMBED_MODEL}")
                _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_collection():
    """获取 ChromaDB collection 单例，使用锁保护初始化"""
    global _chroma_client, _collection
    if _collection is None:
        with _init_lock:
            # 双重检查
            if _collection is None:
                _chroma_client = chromadb.PersistentClient(
                    path=CHROMA_DIR,
                    settings=Settings(anonymized_telemetry=False),
                )
                _collection = _chroma_client.get_or_create_collection(
                    name="knowledge_base",
                    metadata={"hnsw:space": "cosine"},
                )
                log.info(f"ChromaDB 已加载，当前文档数: {_collection.count()}")
    return _collection


# ─── 文档入库 ─────────────────────────────────────────────────────────────────

def ingest_pdf(file_path: str, chunk_size: int = 400, overlap: int = 50) -> int:
    """
    读取 PDF，分块后向量化入库。
    返回成功入库的 chunk 数量。
    
    安全措施：
    - 检查文件扩展名必须是 .pdf
    - 检查文件路径在白名单目录内
    """
    # 安全检查：文件扩展名
    if not file_path.lower().endswith('.pdf'):
        raise ValueError("只支持 PDF 文件")
    
    # 安全检查：路径白名单
    if not _is_safe_path(file_path):
        log.error(f"拒绝访问路径: {file_path}")
        raise PermissionError(f"文件路径不在允许的目录内: {file_path}")
    
    embed  = _get_embed_model()
    col    = _get_collection()

    chunks = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                # 按字符滑动窗口分块
                for i in range(0, len(text), chunk_size - overlap):
                    chunk = text[i : i + chunk_size].strip()
                    if len(chunk) > 30:
                        chunks.append((chunk, f"p{page_num}_c{i}"))
    except Exception as e:
        log.error(f"PDF 解析失败: {e}")
        raise

    if not chunks:
        return 0

    texts = [c[0] for c in chunks]
    ids   = [c[1] for c in chunks]

    # 批量向量化
    vectors = embed.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    # 入库（upsert 避免重复）
    col.upsert(documents=texts, embeddings=vectors, ids=ids)
    log.info(f"PDF 入库完成，共 {len(chunks)} 个 chunk")
    return len(chunks)


def clear_knowledge_base() -> None:
    """清空知识库"""
    col = _get_collection()
    # 先获取所有文档的 ID，然后批量删除
    existing = col.get()
    if existing and existing.get("ids"):
        col.delete(ids=existing["ids"])
        log.info(f"知识库已清空，删除了 {len(existing['ids'])} 个文档")
    else:
        log.info("知识库为空，无需清空")


# ─── 检索 ─────────────────────────────────────────────────────────────────────

def search(query: str) -> str:
    """
    检索与 query 最相关的 Top-K 文档。
    如果知识库为空或无相关结果，返回空字符串。
    返回格式化的【系统检索结果】字符串，直接注入 System Prompt。
    """
    col = _get_collection()
    if col.count() == 0:
        return ""

    embed = _get_embed_model()
    query_vec = embed.encode([query], show_progress_bar=False).tolist()

    try:
        results = col.query(
            query_embeddings=query_vec,
            n_results=min(RAG_TOP_K, col.count()),
            include=["documents", "distances"],
        )
    except Exception as e:
        log.warning(f"RAG 检索失败: {e}")
        return ""

    docs      = results.get("documents", [[]])
    distances = results.get("distances", [[]])
    
    # 安全检查：确保返回结果格式正确
    if not docs or not isinstance(docs, list) or len(docs) == 0:
        return ""
    docs = docs[0] if isinstance(docs[0], list) else docs
    
    if not distances or not isinstance(distances, list) or len(distances) == 0:
        return ""
    distances = distances[0] if isinstance(distances[0], list) else distances
    
    if not docs:
        return ""

    # 相似度过滤（cosine distance < 阈值 认为相关）
    relevant = [
        doc for doc, dist in zip(docs, distances)
        if dist < RAG_DISTANCE_THRESHOLD
    ]
    if not relevant:
        return ""

    context = "\n---\n".join(relevant)
    return f"【系统检索结果】\n{context}"


def kb_count() -> int:
    """返回知识库当前文档数"""
    try:
        return _get_collection().count()
    except Exception:
        return 0
