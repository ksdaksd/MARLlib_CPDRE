---
name: human-write
description: Chinese professional document drafting, rewriting, and formatting standardization with anti-AI tone cleanup and delivery-ready structure checks. Use when Codex needs to create or revise formal Chinese materials such as reports, research plans, policy notes, project proposals, meeting summaries, interview prep docs, evidence compilations, and Word/PPT-ready manuscripts.
---

# Human Write

Deliver formal Chinese documents that read like human-authored work, with stable structure and clean layout for direct delivery or Word/PPT conversion.

Use this skill as the default standard layer before type-specific requirements.

## Workflow

1. Classify the document request before writing.
2. Apply universal writing and layout rules.
3. Add type-specific constraints.
4. Perform anti-AI tone cleanup and evidence checks.
5. Run final acceptance checks for delivery readiness.

For full standards and type-specific rules, load:
`references/writing-standard-zh.md`

## Step 1: Classify Request Type

Identify one primary type:
- Research review
- Research proposal
- Policy/consulting memo
- Cooperation/project proposal
- Evidence compilation
- Meeting minutes/interview summary
- Grant/application material
- Training/course handout
- Generic formal document

If user requests a complete article/proposal/plan, do not collapse into outline-only output.

## Step 2: Apply Universal Rules

Enforce these rules in every output:
- Write direct, formal, natural Chinese.
- Make each paragraph carry one main point.
- State judgment first, then explanation or evidence.
- Prefer specific facts, process, and rationale over slogans.
- Keep headings short and natural; avoid template-like labels.
- Keep terms, abbreviations, and naming consistent.
- Preserve traceability for claims, sources, and references.

## Step 3: Add Type-Specific Constraints

Load the matching section in `references/writing-standard-zh.md` and apply it strictly.

Minimum requirement by type:
- Research-like text: include question, method/design, findings or expected outcomes, and limits.
- Proposal/plan text: include logic for option choice, steps, ownership, risks, and fallback.
- Evidence compilation: include category, core content, usage/insight, and source for each item.
- Meeting/interview summary: separate key points, disagreements, decisions, and action items.

## Step 4: Anti-AI Tone Cleanup

Before finalizing, remove or rewrite:
- Buzzword-heavy management phrasing.
- Mechanical rhetorical templates repeated across sections.
- Empty meta-writing lines that explain the writing process instead of content.
- Generic summaries with no scenario, data, method, or action detail.

Prefer concrete rewrites:
- Replace abstract labels with actionable headings.
- Replace empty conclusions with conclusion plus basis.
- Expand one-line examples into context, process, judgment, and result.

## Step 5: Layout and Delivery Checks

When output targets Word delivery, enforce:
- Chinese body text style consistency.
- First-line indentation and readable spacing.
- Heading hierarchy with clear visual separation.
- Black text styling without accidental theme-color artifacts.

If generating or converting docx, check at least:
- Chinese text displays correctly.
- Main title alignment is correct.
- Heading/body styles are consistent.
- Paragraph formatting is stable across sections.

## Output Contract

When returning final content, ensure:
- Structure is complete and connected.
- Core judgments are explicit and supported.
- Type-specific requirements are satisfied.
- Language is natural and not obviously AI-templated.
- Content can be copied into Word/PPT without structural rework.
