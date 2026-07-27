import { SpanStatusCode } from "@opentelemetry/api"
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
  if (span) {
    if (totals) {
      span.setAttributes({
        "interaction.total_tokens": totals.tokens,
        "interaction.total_cost_usd": totals.cost,
        "interaction.total_messages": totals.messages,
      })
    }
    span.setStatus(error ? { code: status, message: error } : { code: status })
    if (error) span.setAttribute("error", error)
    span.end(endTime)
    ctx.interactionSpans.delete(interactionID)
  }
  ctx.interactionTotals.delete(interactionID)
  if (ctx.activeInteractions.get(sessionID) === interactionID) {
    ctx.activeInteractions.delete(sessionID)
  }
}

export { endInteractionSpan }
