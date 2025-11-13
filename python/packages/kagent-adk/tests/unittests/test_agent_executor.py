"""Test A2aAgentExecutor to verify trace span attributes."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest
from a2a.types import Message, Part, Role, TextPart
from google.adk.runners import Runner
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from kagent.adk._agent_executor import A2aAgentExecutor


@pytest.fixture
def span_exporter():
    """Create an in-memory span exporter to capture spans."""
    exporter = InMemorySpanExporter()
    
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture
def mock_runner():
    """Create a mock runner for testing."""
    runner = MagicMock(spec=Runner)
    runner.app_name = "test_app"
    
    session_service = MagicMock()
    session_service.get_session = AsyncMock(return_value=None)
    session_service.create_session = AsyncMock()
    session_service.append_event = AsyncMock()
    runner.session_service = session_service
    
    mock_session = MagicMock()
    mock_session.id = "test-session-123"
    mock_session.user_id = "test-user"
    session_service.create_session.return_value = mock_session
    
    mock_invocation_context = MagicMock()
    mock_invocation_context.app_name = "test_app"
    runner._new_invocation_context = MagicMock(return_value=mock_invocation_context)
    
    async def mock_run_async(**kwargs):
        if False:
            yield
    
    runner.run_async = mock_run_async
    
    return runner


def create_mock_request_context(
    context_id: str = "test-context-123",
    task_id: str = "test-task-456",
    message_text: str = "Test message",
    user_id: str | None = None,
):
    """Create a mock RequestContext for testing."""
    context = MagicMock()
    context.context_id = context_id
    context.task_id = task_id
    context.current_task = None
    
    message = Message(
        message_id="msg-123",
        role=Role.user,
        parts=[Part(TextPart(text=message_text))],
    )
    context.message = message
    
    call_context = MagicMock()
    call_context.state = {"headers": {}}
    
    if user_id:
        # Simulate an authenticated user by setting the header.
        call_context.state["headers"]["x-user-id"] = user_id
        
        user = MagicMock()
        user.user_name = user_id
        call_context.user = user
    else:
        call_context.user = None
    
    context.call_context = call_context
    
    return context


@pytest.mark.asyncio
async def test_span_attributes_with_user_id(span_exporter, mock_runner):
    """Test that span attributes are set correctly when user_id is present."""
    
    def create_runner():
        return mock_runner
    
    executor = A2aAgentExecutor(runner=create_runner)
    
    # Create a context with an authenticated user_id.
    context = create_mock_request_context(user_id="test-user@example.com")
    
    event_queue = AsyncMock()
    event_queue.enqueue_event = AsyncMock()
    
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span") as span:
        try:
            await executor.execute(context, event_queue)
        except Exception as e:
            print(f"Execution error (expected in test): {e}")
    
    spans = span_exporter.get_finished_spans()
    
    test_span = None
    for s in spans:
        if s.name == "test_span":
            test_span = s
            break
    
    assert test_span is not None, "Test span not found"
    
    attributes = dict(test_span.attributes or {})
    
    print(f"Span attributes: {attributes}")
    
    assert "kagent.user_id" in attributes, "kagent.user_id attribute should be set"
    assert "gen_ai.task.id" in attributes, "gen_ai.task.id attribute should be set"
    assert "gen_ai.converstation.id" in attributes, "gen_ai.converstation.id attribute should be set"
    
    assert attributes["kagent.user_id"] == "test-user@example.com"
    assert attributes["gen_ai.task.id"] == "test-task-456"
    assert attributes["gen_ai.converstation.id"] == "test-context-123"


@pytest.mark.asyncio
async def test_span_attributes_without_user_id(span_exporter, mock_runner):
    """Test that span attributes are set correctly when user_id falls back to A2A_USER pattern."""
    
    def create_runner():
        return mock_runner
    
    executor = A2aAgentExecutor(runner=create_runner)
    
    # Create a context without an authenticated user_id.
    context = create_mock_request_context()
    
    event_queue = AsyncMock()
    event_queue.enqueue_event = AsyncMock()
    
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span") as span:
        try:
            await executor.execute(context, event_queue)
        except Exception as e:
            print(f"Execution error (expected in test): {e}")
    
    spans = span_exporter.get_finished_spans()
    
    test_span = None
    for s in spans:
        if s.name == "test_span":
            test_span = s
            break
    
    assert test_span is not None, "Test span not found"
    
    attributes = dict(test_span.attributes or {})
    
    print(f"Span attributes: {attributes}")
    
    assert "kagent.user_id" in attributes, "kagent.user_id attribute should be set even with fallback user_id"
    assert "gen_ai.task.id" in attributes, "gen_ai.task.id attribute should be set"
    assert "gen_ai.converstation.id" in attributes, "gen_ai.converstation.id attribute should be set"
    
    assert attributes["kagent.user_id"] == "A2A_USER_test-context-123"
    assert attributes["gen_ai.task.id"] == "test-task-456"
    assert attributes["gen_ai.converstation.id"] == "test-context-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
