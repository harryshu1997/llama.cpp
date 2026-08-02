# S37: Arbitrary Resident-Layer Exit

## Question

Can a live request keep one request-level route through prefill and decode while
the phone-to-CUDA handoff is selected at any layer jointly resident around the
handoff?

## Scope

The resident topology is unchanged from S36:

- OP12 and OP15 hold `[0,8)`;
- the selected A6000 terminal worker holds `[4,48)`;
- therefore the complete finite set of legal handoffs is `{4,5,6,7,8}`.

An exit outside that intersection is not legal: at least one endpoint lacks a
required weight. S37 does not claim arbitrary placement across all 48 layers.

## Physical Gate

For both phones and every legal cut:

1. run two B8 request groups through prompt plus four terminal outputs;
2. pin one cut for the full request lifetime;
3. observe phone range `[0,cut)` and tail range `[cut,48)`;
4. require identical terminal tokens across every route;
5. require zero active sequence, route pin, and software lease after each group;
6. detach without unloading resident weights.

This follows the S36 mixed-phase proof. No latency, energy, or scheduler-benefit
claim is gated here.
