import logging
import sys
import os
import json
import threading
from logging import Logger, StreamHandler
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pythonjsonlogger import jsonlogger
from typing import Literal, Optional

# Globally shared error handler, used by all answer loggers
_shared_error_handler = None
_handler_lock = threading.Lock()


class ColoredStructuredFormatter(logging.Formatter):
    """Colored structured log formatter."""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'RESET': '\033[0m'
    }

    def format(self, record):
        # Use a special format for verification operations
        if hasattr(record, 'op_id'):
            op_id = record.op_id
            level_color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']

            # Build main message - remove duplicate levelname
            msg_parts = [
                f"{level_color}[{op_id}]{reset}"
            ]

            # Add node info
            if hasattr(record, 'node_id') and record.node_id:
                msg_parts.append(f"Node({record.node_id})")

            # Add verification type
            if hasattr(record, 'verify_type'):
                msg_parts.append(f"<{record.verify_type}>")

            # Add main message
            msg_parts.append(record.getMessage())

            # Build detailed info (indented display)
            details = []

            if hasattr(record, 'node_desc') and record.node_desc:
                details.append(f"  📋 Description: {record.node_desc}")

            if hasattr(record, 'url') and record.url:
                details.append(f"  🔗 URL: {record.url}")

            if hasattr(record, 'claim_preview'):
                details.append(f"  💬 Claim: {record.claim_preview}")

            if hasattr(record, 'reasoning') and record.reasoning:
                reasoning = record.reasoning
                # if len(reasoning) > 200:
                #     reasoning = reasoning[:200] + "..."
                details.append(f"  💭 Reasoning: {reasoning}")

            if hasattr(record, 'result'):
                result_str = "✅ PASS" if record.result else "❌ FAIL"
                details.append(f"  📊 Result: {result_str}")

            # Combine all parts
            full_msg = " ".join(msg_parts)
            if details:
                full_msg += "\n" + "\n".join(details)

            return full_msg

        # For other logs, use standard format - show level only for ERROR/WARNING
        level_indicator = ""
        if record.levelname == 'ERROR':
            level_indicator = f"{self.COLORS['ERROR']}[ERROR]{self.COLORS['RESET']} "
        elif record.levelname == 'WARNING':
            level_indicator = f"{self.COLORS['WARNING']}[WARN]{self.COLORS['RESET']} "

        return f"{level_indicator}{record.getMessage()}"


class ErrorWithContextFormatter(logging.Formatter):
    """Formatter specialized for errors, adding context information."""

    COLORS = {
        'ERROR': '\033[31m',  # Red
        'WARNING': '\033[33m',  # Yellow
        'RESET': '\033[0m'
    }

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']

        # Build context information
        context_parts = []

        # Add agent and answer information
        if hasattr(record, 'agent_name') and record.agent_name:
            context_parts.append(f"Agent:{record.agent_name}")
        if hasattr(record, 'answer_name') and record.answer_name:
            context_parts.append(f"Answer:{record.answer_name}")
        if hasattr(record, 'node_id') and record.node_id:
            context_parts.append(f"Node:{record.node_id}")
        if hasattr(record, 'op_id') and record.op_id:
            context_parts.append(f"Op:{record.op_id}")

        context_str = " | ".join(context_parts)
        context_prefix = f"[{context_str}] " if context_str else ""

        return f"{level_color}[{record.levelname}]{reset} {context_prefix}{record.getMessage()}"


class HumanReadableFormatter(logging.Formatter):
    """Human-readable file log format, keep emojis."""

    def format(self, record):
        # Timestamp - second precision
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')

        # 基本信息 - 只在重要级别显示level
        level_prefix = ""
        if record.levelname in ['ERROR', 'WARNING']:
            level_prefix = f"[{record.levelname}] "

        base_info = f"[{timestamp}] {level_prefix}{record.getMessage()}"

        # Add structured fields
        extras = []
        skip_fields = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
            'filename', 'module', 'lineno', 'funcName', 'created',
            'msecs', 'relativeCreated', 'thread', 'threadName',
            'processName', 'process', 'getMessage', 'exc_info',
            'exc_text', 'stack_info', 'message'
        }

        for key, value in record.__dict__.items():
            if key not in skip_fields and value is not None:
                # Special handling for some fields
                if key == 'final_score' and isinstance(value, (int, float)):
                    extras.append(f"score={value}")
                elif key == 'agent_name':
                    extras.append(f"agent={value}")
                elif key == 'node_id':
                    extras.append(f"node={value}")
                elif key == 'op_id':
                    extras.append(f"op={value}")
                else:
                    extras.append(f"{key}={value}")

        if extras:
            base_info += f" | {' | '.join(extras)}"

        return base_info


class CompactJsonFormatter(jsonlogger.JsonFormatter):
    """Compact JSON formatter that removes redundant fields."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        # Remove unnecessary fields
        fields_to_remove = ['name', 'levelname']
        for field in fields_to_remove:
            log_record.pop(field, None)

        # Simplify time format to seconds
        if 'asctime' in log_record:
            try:
                asctime = log_record['asctime']
                if ',' in asctime:
                    log_record['asctime'] = asctime.split(',')[0]
            except:
                pass


def _get_shared_error_handler() -> StreamHandler:
    """Get or create the globally shared error handler."""
    global _shared_error_handler

    with _handler_lock:
        if _shared_error_handler is None:
            _shared_error_handler = StreamHandler(sys.stderr)  # Use stderr for errors
            _shared_error_handler.setFormatter(ErrorWithContextFormatter())
            _shared_error_handler.setLevel(logging.ERROR)  # Show only ERROR level

    return _shared_error_handler


def create_logger(
        lgr_nm: str,
        log_folder: str,
        enable_console: bool = True,
        file_format: Literal["jsonl", "readable", "both"] = "both",
        enable_shared_errors: bool = False  # New parameter
) -> tuple[Logger, str]:
    """
    Create an independent logger instance, supporting multiple file formats.

    Args:
        lgr_nm: Logger name
        log_folder: Log folder
        enable_console: Whether to enable console output
        file_format: File log format
        enable_shared_errors: Whether to output ERROR-level logs to the shared terminal

    Returns:
        (logger instance, timestamp)
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a unique logger name to avoid duplication
    unique_logger_name = f"{lgr_nm}_{current_time}_{id(log_folder)}"

    # If a logger already exists, clean it up first
    existing_logger = logging.getLogger(unique_logger_name)
    if existing_logger.handlers:
        for handler in existing_logger.handlers[:]:
            existing_logger.removeHandler(handler)
            handler.close()

    # Create a new logger
    new_logger = logging.getLogger(unique_logger_name)
    new_logger.setLevel(logging.DEBUG)
    new_logger.propagate = False

    # File handlers
    if file_format in ["jsonl", "both"]:
        # JSON Lines format
        jsonl_file = os.path.join(log_folder, f"{current_time}_{lgr_nm}.jsonl")
        jsonl_handler = TimedRotatingFileHandler(
            jsonl_file,
            when="D",
            backupCount=14,
            encoding="utf-8"
        )
        jsonl_formatter = CompactJsonFormatter('%(asctime)s %(message)s')
        jsonl_handler.setFormatter(jsonl_formatter)
        jsonl_handler.setLevel(logging.DEBUG)
        new_logger.addHandler(jsonl_handler)

    if file_format in ["readable", "both"]:
        # Human-readable format
        readable_file = os.path.join(log_folder, f"{current_time}_{lgr_nm}.log")
        readable_handler = TimedRotatingFileHandler(
            readable_file,
            when="D",
            backupCount=14,
            encoding="utf-8"
        )
        readable_formatter = HumanReadableFormatter()
        readable_handler.setFormatter(readable_formatter)
        readable_handler.setLevel(logging.DEBUG)
        new_logger.addHandler(readable_handler)

    # Console handler - use colored structured format
    if enable_console:
        console_handler = StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredStructuredFormatter())
        console_handler.setLevel(logging.INFO)
        new_logger.addHandler(console_handler)

    # Shared error handler - for displaying errors during parallel execution
    if enable_shared_errors:
        shared_error_handler = _get_shared_error_handler()
        new_logger.addHandler(shared_error_handler)

    return new_logger, current_time


def create_sub_logger(parent_logger: Logger, sub_name: str) -> Logger:
    """
    Create sublogger based on parent logger, inherit parent logger's handlers
    Used to create hierarchical logs within the same evaluation
    """
    parent_name = parent_logger.name
    sub_logger_name = f"{parent_name}.{sub_name}"

    sub_logger = logging.getLogger(sub_logger_name)
    sub_logger.setLevel(parent_logger.level)
    sub_logger.propagate = True  # Allow propagation to parent logger

    return sub_logger


def cleanup_logger(logger: Logger) -> None:
    """Clean up all handlers of the logger (but not the shared error handler)."""
    global _shared_error_handler

    for handler in logger.handlers[:]:
        # Do not clean up the shared error handler
        if handler is not _shared_error_handler:
            logger.removeHandler(handler)
            handler.close()
        else:
            logger.removeHandler(handler)  # Remove only, do not close


def cleanup_shared_error_handler():
    """Clean up the shared error handler at program end."""
    global _shared_error_handler

    with _handler_lock:
        if _shared_error_handler is not None:
            _shared_error_handler.close()
            _shared_error_handler = None


# Usage examples and notes
"""
How to use in the evaluation runner:

1. Main logger — normal console output:
   main_logger, timestamp = create_logger("main_task", log_folder, enable_console=True)

2. Per-answer loggers — errors are shown in the terminal:
   logger, timestamp = create_logger(
       log_tag, 
       str(log_dir), 
       enable_console=False,  # Do not enable regular console output
       enable_shared_errors=True  # Enable shared error output
   )

This results in:
- Primary progress information shown in the main terminal
- Each answer's ERROR-level messages also shown in the terminal (with context)
- All detailed logs still saved to their respective files

Example terminal output:
🚀 Starting concurrent evaluation of 10 answers
👉 Processing human/answer_1.md
[ERROR] [Agent:human | Answer:answer_1.md | Node:price_check] Failed to verify price claim
👉 Processing openai_deep_research/answer_1.md
✅ Successfully evaluated human/answer_1.md
"""
