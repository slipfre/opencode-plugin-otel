import { SpanStatusCode } from "@opentelemetry/api"
import {
  MimeType,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
} from "@arizeai/openinference-semantic-conventions"
import type { HandlerContext } from "./types.ts"

function endInteractionSpan(
  interactionID: string,
  sessionID: string,
  status: SpanStatusCode.OK | SpanStatusCode.ERROR,
  ctx: HandlerContext,
  endTime?: number,
  error?: string,
) {
  const span = ctx.interactionSpans.get(interactionID)
  const totals = ctx.interactionTotals.get(interactionID)
  const completion = ctx.interactionCompletions.get(interactionID)
  if (span) {
    if (totals) {
      span.setAttributes({
        "interaction.total_tokens": totals.tokens,
        "interaction.total_cost_usd": totals.cost,
        "interaction.total_messages": totals.messages,
      })
    }
    if (completion?.output !== undefined) {
      span.setAttributes({
        [OUTPUT_VALUE]: completion.output,
        [OUTPUT_MIME_TYPE]: MimeType.TEXT,
      })
      const run = ctx.activeRunSpans.get(sessionID)
      const interactionIO = run?.interactionIO.get(interactionID)
      if (run && interactionIO) {
        run.interactionIO.set(interactionID, { ...interactionIO, output: completion.output })
      }
    }
    span.setStatus(error ? { code: status, message: error } : { code: status })
    if (error) span.setAttribute("error", error)
    span.end(endTime)
    ctx.interactionSpans.delete(interactionID)
  }
  ctx.interactionTotals.delete(interactionID)
  ctx.interactionCompletions.delete(interactionID)
  if (ctx.activeInteractions.get(sessionID) === interactionID) {
    ctx.activeInteractions.delete(sessionID)
  }
}

export { endInteractionSpan }
