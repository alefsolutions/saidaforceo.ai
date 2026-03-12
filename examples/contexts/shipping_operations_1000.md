# Dataset: Shipping Operations

## Source Summary
Shipment records across warehouses, routes, and carrier service levels.

## Metric Definitions
parcel_count = number of parcels in the shipment
shipping_cost = total shipment cost in local currency
delay_minutes = delivery delay in minutes compared to the planned schedule

## Business Rules
- shipped_at is the trusted shipment date
- delay_minutes of zero means on-time or early shipment handling

## Trusted Date Fields
- shipped_at

## Preferred Identifiers
- carrier
- route
- service_level
- warehouse
