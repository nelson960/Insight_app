from __future__ import annotations

import ctypes
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
MODEL_VALIDATION_VERSION = "v3"


@dataclass
class ChatTemplateValidation:
    ok: bool
    reason: str
    template: Optional[str] = None
    template_name: Optional[str] = None
    template_variants: Dict[str, str] = field(default_factory=dict)
    rendered_preview: Optional[str] = None
    system_supported: Optional[bool] = None
    stop_token_ids: List[int] = field(default_factory=list)
    stop_token_texts: List[str] = field(default_factory=list)
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None
    eot_id: Optional[int] = None
    add_bos: Optional[bool] = None
    add_eos: Optional[bool] = None


def _get_llama_cpp_lowlevel():
    try:
        from llama_cpp import llama_cpp as ll  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("llama_cpp low-level bindings unavailable") from exc
    return ll


def _get_llama_cpp_versions() -> Tuple[str, str]:
    version = "unknown"
    build_id = "unknown"
    try:
        import llama_cpp  # type: ignore

        version = str(getattr(llama_cpp, "__version__", "unknown"))
    except Exception:
        pass
    try:
        ll = _get_llama_cpp_lowlevel()
        build_id = str(
            getattr(ll, "LLAMA_CPP_BUILD", None)
            or getattr(ll, "LLAMA_BUILD_NUMBER", None)
            or getattr(ll, "LLAMA_BUILD_ID", None)
            or "unknown"
        )
    except Exception:
        pass
    return version, build_id


def compute_model_fingerprint(path: Path) -> str:
    try:
        st = path.stat()
        payload = f"{path}:{st.st_size}:{st.st_mtime_ns}"
    except Exception:
        payload = str(path)
    try:
        import hashlib

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        return payload


def build_model_record(path: Path, validation: ChatTemplateValidation) -> Dict[str, Any]:
    try:
        st = path.stat()
        size_bytes = int(st.st_size)
        mtime = float(st.st_mtime)
    except Exception:
        size_bytes = 0
        mtime = 0.0
    fingerprint = compute_model_fingerprint(path)
    llama_cpp_version, llama_cpp_build_id = _get_llama_cpp_versions()
    return {
        "model_id": fingerprint,
        "path": str(path),
        "size_bytes": size_bytes,
        "mtime": mtime,
        "llama_cpp_python_version": llama_cpp_version,
        "llama_cpp_build_id": llama_cpp_build_id,
        "validation_version": MODEL_VALIDATION_VERSION,
        "accepted": bool(validation.ok),
        "reason": validation.reason,
        "template_selected": validation.template,
        "template_selected_name": validation.template_name,
        "template_variants": validation.template_variants,
        "template_system_supported": validation.system_supported,
        "stop_token_ids": validation.stop_token_ids,
        "stop_token_texts": validation.stop_token_texts,
        "bos_id": validation.bos_id,
        "eos_id": validation.eos_id,
        "eot_id": validation.eot_id,
        "add_bos": validation.add_bos,
        "add_eos": validation.add_eos,
    }


def model_record_is_current(record: Optional[Dict[str, Any]], path: Path) -> bool:
    if not record or not isinstance(record, dict):
        return False
    if record.get("accepted") is not True:
        return False
    if str(record.get("path") or "") != str(path):
        return False
    if str(record.get("validation_version") or "") != MODEL_VALIDATION_VERSION:
        return False
    try:
        if str(record.get("model_id") or "") != compute_model_fingerprint(path):
            return False
    except Exception:
        return False
    llama_cpp_version, llama_cpp_build_id = _get_llama_cpp_versions()
    current_version = str(llama_cpp_version or "")
    current_build = str(llama_cpp_build_id or "")
    # Only enforce version/build matches when we can reliably read them.
    if current_version and current_version != "unknown":
        if str(record.get("llama_cpp_python_version") or "") != current_version:
            return False
    if current_build and current_build != "unknown":
        if str(record.get("llama_cpp_build_id") or "") != current_build:
            return False
    return True


def _parse_template_json(raw: str) -> Optional[Dict[str, str]]:
    s = (raw or "").strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    out: Dict[str, str] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, str) and v.strip():
            out[k] = v
    return out or None


def extract_chat_templates(metadata: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, str]]:
    default_template = None
    variants: Dict[str, str] = {}
    try:
        raw_default = metadata.get("tokenizer.chat_template")
        if isinstance(raw_default, str) and raw_default.strip():
            default_template = raw_default
    except Exception:
        default_template = None

    # Variant keys (tokenizer.chat_template.NAME)
    for k, v in (metadata or {}).items():
        if not isinstance(k, str):
            continue
        if not k.startswith("tokenizer.chat_template."):
            continue
        name = k.split("tokenizer.chat_template.", 1)[1].strip()
        if not name:
            continue
        if isinstance(v, str) and v.strip():
            variants[name] = v

    # JSON bundle (if default is JSON)
    if isinstance(default_template, str):
        json_templates = _parse_template_json(default_template)
        if json_templates:
            for k, v in json_templates.items():
                variants.setdefault(k, v)
            if "default" in json_templates:
                default_template = json_templates["default"]

    return default_template, variants


def select_chat_template(
    default_template: Optional[str], variants: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(default_template, str) and default_template.strip():
        return "default", default_template
    if variants:
        name = sorted(variants.keys())[0]
        return name, variants[name]
    return None, None


def normalize_messages_for_template(
    messages: List[Dict[str, str]],
    *,
    default_system: str = "",
) -> List[Dict[str, str]]:
    normalized = list(messages or [])
    if not normalized or normalized[0].get("role") != "system":
        normalized.insert(0, {"role": "system", "content": default_system})
    return normalized


def apply_chat_template_minja(
    template: str,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    ll = _get_llama_cpp_lowlevel()
    if not hasattr(ll, "llama_chat_apply_template") or not hasattr(ll, "llama_chat_message"):
        raise RuntimeError("llama_chat_apply_template unavailable in llama_cpp")

    msg_type = ll.llama_chat_message  # type: ignore[attr-defined]
    count = len(messages)
    arr = (msg_type * count)()  # type: ignore[misc]
    # Keep buffers alive for the duration of the call.
    keepalive: List[bytes] = []
    for i, msg in enumerate(messages):
        role = (msg.get("role") or "").encode("utf-8")
        content = (msg.get("content") or "").encode("utf-8")
        keepalive.append(role)
        keepalive.append(content)
        arr[i].role = role
        arr[i].content = content

    tmpl_b = template.encode("utf-8")

    def _call(buf, buf_size: int) -> int:
        try:
            return int(ll.llama_chat_apply_template(tmpl_b, arr, count, add_generation_prompt, buf, buf_size))
        except TypeError:
            return int(ll.llama_chat_apply_template(arr, count, add_generation_prompt, buf, buf_size))

    # Start with a reasonable buffer; expand if needed.
    buf_size = max(4096, len(template) * 2 + 256)
    for _ in range(6):
        buf = ctypes.create_string_buffer(buf_size)
        res = _call(buf, buf_size)
        if res < 0:
            buf_size *= 2
            continue
        if res >= buf_size:
            buf_size = res + 1
            continue
        return buf.value.decode("utf-8", errors="ignore")
    raise RuntimeError("llama_chat_apply_template failed to render within buffer limits")


def _detect_multimodal(llm, model_path: Optional[Path]) -> bool:
    ll = None
    try:
        ll = _get_llama_cpp_lowlevel()
    except Exception:
        ll = None

    if ll is not None and hasattr(ll, "llama_model_has_encoder"):
        try:
            model_ptr = getattr(llm, "_model", None)
            if model_ptr is not None and bool(ll.llama_model_has_encoder(model_ptr)):
                return True
        except Exception:
            pass

    if model_path is not None:
        try:
            for fn in os.listdir(model_path.parent):
                low = fn.lower()
                if low.startswith("mmproj") and low.endswith(".gguf"):
                    return True
                if "vision" in low or "clip" in low:
                    if low.endswith(".gguf"):
                        return True
        except Exception:
            pass
    return False


def _extract_special_tokens(llm) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[bool], Optional[bool], List[int], List[str]]:
    ll = None
    try:
        ll = _get_llama_cpp_lowlevel()
    except Exception:
        ll = None

    bos_id = eos_id = eot_id = None
    add_bos = add_eos = None
    stop_ids: List[int] = []
    stop_texts: List[str] = []

    if ll is not None:
        try:
            model_ptr = getattr(llm, "_model", None)
            if model_ptr is not None and hasattr(ll, "llama_model_get_vocab"):
                vocab = ll.llama_model_get_vocab(model_ptr)
                if hasattr(ll, "llama_vocab_bos"):
                    bos_id = int(ll.llama_vocab_bos(vocab))
                if hasattr(ll, "llama_vocab_eos"):
                    eos_id = int(ll.llama_vocab_eos(vocab))
                if hasattr(ll, "llama_vocab_eot"):
                    eot_id = int(ll.llama_vocab_eot(vocab))
                if hasattr(ll, "llama_vocab_get_add_bos"):
                    add_bos = bool(ll.llama_vocab_get_add_bos(vocab))
                if hasattr(ll, "llama_vocab_get_add_eos"):
                    add_eos = bool(ll.llama_vocab_get_add_eos(vocab))
        except Exception:
            pass

    for tid in (eot_id, eos_id):
        if tid is None:
            continue
        if int(tid) >= 0:
            stop_ids.append(int(tid))

    stop_ids = sorted(set(stop_ids))

    try:
        for tid in stop_ids:
            piece = llm.detokenize([int(tid)]).decode("utf-8", errors="ignore")
            if piece:
                stop_texts.append(piece)
    except Exception:
        stop_texts = []

    return bos_id, eos_id, eot_id, add_bos, add_eos, stop_ids, stop_texts


def validate_chat_template_for_llm(
    llm,
    *,
    model_path: Optional[Path] = None,
    require_chat_template: bool = True,
    reject_multimodal: bool = True,
) -> ChatTemplateValidation:
    try:
        metadata = getattr(llm, "metadata", {}) or {}
    except Exception:
        metadata = {}

    default_template, variants = extract_chat_templates(metadata)
    if require_chat_template and not default_template and not variants:
        return ChatTemplateValidation(ok=False, reason="missing_chat_template", template_variants=variants)

    if reject_multimodal and _detect_multimodal(llm, model_path):
        return ChatTemplateValidation(ok=False, reason="multimodal_model_detected", template_variants=variants)

    candidates: List[Tuple[str, str]] = []
    if isinstance(default_template, str) and default_template.strip():
        candidates.append(("default", default_template))
    for name, tmpl in variants.items():
        if isinstance(tmpl, str) and tmpl.strip():
            candidates.append((name, tmpl))

    rendered = None
    selected_name = None
    selected_template = None
    system_supported = False
    system_probe = "__INSIGHT_SYSTEM_TEST__"
    preview_messages = [
        {"role": "system", "content": system_probe},
        {"role": "user", "content": "Hello"},
    ]
    for name, tmpl in candidates:
        try:
            rendered_try = apply_chat_template_minja(
                tmpl,
                preview_messages,
                add_generation_prompt=True,
            )
            if rendered_try and rendered_try.strip():
                rendered = rendered_try
                selected_name = name
                selected_template = tmpl
                system_supported = system_probe in rendered_try
                break
        except Exception as exc:
            logger.debug("Template validation failed name=%s err=%s", name, exc)
            continue

    if require_chat_template and not rendered:
        return ChatTemplateValidation(ok=False, reason="chat_template_apply_failed", template_variants=variants)
    if rendered and not system_supported:
        # Allow templates that ignore system messages, but record the flag so the
        # runtime can safely inject the system content into the user turn.
        logger.warning("Chat template does not preserve system messages; enabling system-to-user fallback")

    bos_id, eos_id, eot_id, add_bos, add_eos, stop_ids, stop_texts = _extract_special_tokens(llm)

    return ChatTemplateValidation(
        ok=True,
        reason="ok",
        template=selected_template,
        template_name=selected_name,
        template_variants=variants,
        rendered_preview=rendered,
        system_supported=system_supported,
        stop_token_ids=stop_ids,
        stop_token_texts=stop_texts,
        bos_id=bos_id,
        eos_id=eos_id,
        eot_id=eot_id,
        add_bos=add_bos,
        add_eos=add_eos,
    )


def validate_chat_template_for_path(
    path: Path,
    *,
    require_chat_template: bool = True,
    reject_multimodal: bool = True,
) -> ChatTemplateValidation:
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("llama_cpp is required to validate chat templates") from exc

    llm = Llama(model_path=str(path), vocab_only=True, verbose=False)
    return validate_chat_template_for_llm(
        llm,
        model_path=path,
        require_chat_template=require_chat_template,
        reject_multimodal=reject_multimodal,
    )


def chat_template_error_message(reason: str) -> str:
    reason = (reason or "").strip().lower()
    if reason == "missing_chat_template":
        return "This GGUF does not include an embedded chat template."
    if reason == "multimodal_model_detected":
        return "Multimodal models are not supported yet. Please choose a text-only GGUF."
    if reason == "chat_template_apply_failed":
        return "The embedded chat template is not compatible with llama.cpp's minja engine."
    if reason == "system_prompt_ignored":
        return "The embedded chat template does not preserve the system message. Choose a model that supports system prompts."
    if reason == "llama_cpp_unavailable":
        return "llama_cpp is required to validate chat templates."
    return "Unsupported chat template for this model."


__all__ = [
    "ChatTemplateValidation",
    "MODEL_VALIDATION_VERSION",
    "extract_chat_templates",
    "select_chat_template",
    "normalize_messages_for_template",
    "apply_chat_template_minja",
    "validate_chat_template_for_llm",
    "validate_chat_template_for_path",
    "chat_template_error_message",
    "build_model_record",
    "compute_model_fingerprint",
    "model_record_is_current",
]
