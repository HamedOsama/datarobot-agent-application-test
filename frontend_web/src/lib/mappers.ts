import { v4 as uuid } from 'uuid';
import type {
  Message,
  ReasoningMessageChunkEvent,
  ReasoningMessageContentEvent,
  ReasoningMessageEndEvent,
  ReasoningMessageStartEvent,
  TextMessageChunkEvent,
  TextMessageContentEvent,
  TextMessageEndEvent,
  TextMessageStartEvent,
  ToolCallEndEvent,
} from '@ag-ui/core';
import { EventType } from '@ag-ui/core';
import type { ChatStateEventByType } from '@/types/events';
import { MessageResponse } from '@/api/chat/types.ts';
import { ToolInvocationUIPart } from '@/types/message.ts';

type AgUiTextEvent =
  | TextMessageStartEvent
  | TextMessageContentEvent
  | TextMessageEndEvent
  | TextMessageChunkEvent;

type AgUiReasoningEvent =
  | ReasoningMessageStartEvent
  | ReasoningMessageContentEvent
  | ReasoningMessageChunkEvent
  | ReasoningMessageEndEvent;

type AgUiToolEvent = ToolCallEndEvent;

export function createTextMessageFromAgUiEvent(
  event: AgUiTextEvent,
  textMessageBuffer?: string
): MessageResponse {
  const baseMessage: MessageResponse = {
    id: event.messageId || '',
    content: {
      format: 2,
      parts: [],
      content: '',
    },
    role: 'assistant',
    createdAt: event.timestamp ? new Date(event.timestamp) : new Date(),
    threadId: '',
    resourceId: '',
  };

  // Map role, converting 'developer' to 'system' for compatibility
  const mapRole = (
    role?: 'user' | 'assistant' | 'system' | 'developer'
  ): 'user' | 'assistant' | 'system' => {
    if (!role || role === 'developer') return 'system';
    return role;
  };

  switch (event.type) {
    case EventType.TEXT_MESSAGE_START:
      return {
        ...baseMessage,
        role: mapRole(event.role),
        id: event.messageId,
      };

    case EventType.TEXT_MESSAGE_CONTENT:
      return {
        ...baseMessage,
        id: event.messageId,
        content: {
          format: 2,
          parts: [
            {
              type: 'text',
              text: textMessageBuffer + event.delta,
            },
          ],
          content: textMessageBuffer + event.delta,
        },
      };

    case EventType.TEXT_MESSAGE_END:
      return {
        ...baseMessage,
        id: event.messageId,
      };

    case EventType.TEXT_MESSAGE_CHUNK:
      return {
        ...baseMessage,
        id: event.messageId || '',
        role: mapRole(event.role),
        content: {
          format: 2,
          parts: event.delta
            ? [
                {
                  type: 'text',
                  text: event.delta,
                },
              ]
            : [],
          content: event.delta || '',
        },
      };

    default:
      return baseMessage;
  }
}

export function createToolMessageFromAgUiEvent(
  event: AgUiToolEvent,
  toolCallName: string,
  toolCallArgs: Record<string, any>
): MessageResponse {
  const baseMessage: MessageResponse = {
    id: uuid(),
    content: {
      format: 2,
      parts: [
        {
          type: 'tool-invocation',
          toolInvocation: {
            state: 'call',
            args: toolCallArgs || event.rawEvent?.args || {},
            toolCallId: event.toolCallId,
            toolName: toolCallName,
          } as any,
        },
      ],
    },
    role: 'assistant',
    createdAt: event.timestamp ? new Date(event.timestamp) : new Date(),
    threadId: '',
    resourceId: uuid(),
  };

  return baseMessage;
}

export function createTextMessageFromUserInput({
  message,
  chatId,
  messageId,
}: {
  message: string;
  chatId: string;
  messageId: string;
}): MessageResponse {
  const baseMessage: MessageResponse = {
    id: messageId,
    content: {
      format: 2,
      parts: [
        {
          type: 'text',
          text: message,
        },
      ],
      content: message,
    },
    role: 'user',
    createdAt: new Date(),
    threadId: chatId,
    resourceId: uuid(),
  };

  return baseMessage;
}

export function createCustomMessageWidget({
  toolCallName,
  toolCallArgs,
  threadId,
}: {
  toolCallName: string;
  toolCallArgs: Record<string, any>;
  threadId: string;
}): MessageResponse {
  const toolInvocation = {
    state: 'call',
    toolCallId: `call_${uuid()}`,
    toolName: toolCallName,
    args: toolCallArgs,
  };
  return {
    id: uuid(),
    content: {
      format: 2,
      parts: [
        {
          type: 'tool-invocation',
          toolInvocation,
        },
      ],
    },
    role: 'assistant',
    createdAt: new Date(),
    threadId: threadId,
    resourceId: uuid(),
  };
}

export function messageToStateEvent(message: MessageResponse): ChatStateEventByType<'message'> {
  return {
    type: 'message',
    value: message,
  };
}

export function messageResponseToAgUiMessage(message: MessageResponse): Message | null {
  const { id, role, content } = message;

  if (role === 'user' || role === 'system') {
    const textContent = content.content ?? content.parts.find(p => p.type === 'text')?.text ?? '';
    return { id, role, content: textContent };
  }

  if (role === 'assistant') {
    const toolInvocationParts = content.parts.filter(p => p.type === 'tool-invocation') as ToolInvocationUIPart[];
    if (toolInvocationParts.length > 0) {
      return {
        id,
        role: 'assistant',
        toolCalls: toolInvocationParts.map(p => ({
          id: p.toolInvocation.toolCallId ?? id,
          type: 'function' as const,
          function: {
            name: p.toolInvocation.toolName,
            arguments: typeof p.toolInvocation.args === 'string'
              ? p.toolInvocation.args
              : JSON.stringify(p.toolInvocation.args ?? {}),
          },
        })),
      };
    }
    const textContent = content.content ?? content.parts.find(p => p.type === 'text')?.text ?? '';
    return { id, role: 'assistant', content: textContent };
  }

  if (role === 'reasoning') {
    const reasoningContent = content.content ?? '';
    return { id, role: 'reasoning', content: reasoningContent };
  }

  return null;
}

function reasoningPart(reasoningText: string): {
  type: 'reasoning';
  reasoning: string;
  details: Array<{ type: 'text'; text: string }>;
} {
  return {
    type: 'reasoning',
    reasoning: reasoningText,
    details: reasoningText ? [{ type: 'text', text: reasoningText }] : [],
  };
}

export function createReasoningMessage(
  event: AgUiReasoningEvent,
  reasoningMessageBuffer = ''
): MessageResponse {
  const reasoningMessage: MessageResponse = {
    id: event.messageId ?? uuid(),
    role: 'reasoning',
    content: {
      format: 2,
      parts: [],
      content: '',
    },
    createdAt: new Date(),
    threadId: '',
    resourceId: uuid(),
  };

  switch (event.type) {
    case EventType.REASONING_MESSAGE_START:
      return { ...reasoningMessage, id: event.messageId ?? reasoningMessage.id };
    case EventType.REASONING_MESSAGE_CONTENT: {
      const fullContent = reasoningMessageBuffer + event.delta;
      return {
        ...reasoningMessage,
        id: event.messageId ?? reasoningMessage.id,
        content: {
          format: 2,
          parts: [reasoningPart(fullContent)],
          content: fullContent,
        },
      };
    }
    case EventType.REASONING_MESSAGE_END:
      return { ...reasoningMessage, id: event.messageId ?? reasoningMessage.id };

    case EventType.REASONING_MESSAGE_CHUNK: {
      return {
        ...reasoningMessage,
        id: event.messageId ?? '',
        content: {
          format: 2,
          parts: event.delta ? [reasoningPart(event.delta)] : [],
          content: event.delta,
        },
      };
    }
    default:
      return reasoningMessage;
  }
}
