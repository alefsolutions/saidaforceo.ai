![SAIDA Banner](../../assets/github-banner.png)

# Dataset: Support Tickets

## Source Summary
Daily support ticket records across multiple teams, priorities, and product areas.

## Metric Definitions
resolution_hours = elapsed hours from ticket creation to resolution
csat_score = post-resolution customer satisfaction score on a five point scale

## Business Rules
- reopened_flag yes means the original ticket required additional work
- created_at is the trusted ticket creation date

## Caveats
- csat_score is not available for every real-world ticket workflow, but this example treats it as complete

## Trusted Date Fields
- created_at

## Preferred Identifiers
- team
- priority
- channel
- product_area

