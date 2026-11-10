# Saferproduct dataset

Source: [SaferProducts.gov Public Search](https://www.saferproducts.gov/PublicSearch) — full-database download button (no date or category filters available).

## Incident Counts per Category and Severity

| Category | Sev. 1 | Sev. 2 | Sev. 3 | Sev. 4 | Total |
|-----------|--------|--------|--------|--------|--------|
| Baby | 281 | 1678 | 1649 | 95 | 3703 |
| Clothing and Accessories | 79 | 346 | 1386 | 1 | 1812 |
| Containers & Packaging | 143 | 424 | 380 | 2 | 949 |
| Drywall | 4 | 18 | 42 | 0 | 64 |
| Electronics | 203 | 2885 | 1592 | 5 | 4685 |
| Fuel, Lighters and Fireworks | 25 | 297 | 175 | 3 | 500 |
| Furniture, Furnishings & Decorations | 328 | 3045 | 1957 | 72 | 5402 |
| Hobby | 32 | 169 | 154 | 1 | 356 |
| Home Maintenance and Structures | 460 | 6927 | 2993 | 47 | 10427 |
| Kitchen | 867 | 17221 | 4603 | 9 | 22700 |
| Personal Care | 30 | 520 | 479 | 0 | 1029 |
| Products at Public Facilities | 6 | 40 | 43 | 1 | 90 |
| Sports and Recreation | 207 | 1664 | 1968 | 76 | 3915 |
| Toys & Children | 231 | 1473 | 1454 | 8 | 3166 |
| Yard & Garden | 154 | 2133 | 885 | 18 | 3190 |

## Global Stats

- Total records: 61988
- Product categories: 15

## Time Period

Date range of the raw dump (`data/raw/saferproducts_gov/IncidentReports.csv`, 63,720 reports, no missing dates):

| Field | Earliest | Latest |
|-------|----------|--------|
| Report Date | 2011-03-11 | 2025-09-14 |
| Publication Date | 2011-04-01 | 2025-10-01 |

The download is a snapshot of everything published on the site at download time, so its cut-off is defined by **Publication Date** (a fresh download on 2026-11-08 ends at Publication Date 2026-11-06 — the last business day — while Report Date ends ~3 weeks earlier, since reports are published after review). Report Date is the field to use for the period the incidents were reported.

## Average Text Length (by Category)

| Category | Avg. Length |
|-----------|-------------|
| Baby | 607.2 |
| Clothing and Accessories | 651.3 |
| Containers & Packaging | 687.2 |
| Drywall | 1024.5 |
| Electronics | 637.3 |
| Fuel, Lighters and Fireworks | 733.6 |
| Furniture, Furnishings & Decorations | 688.3 |
| Hobby | 811.6 |
| Home Maintenance and Structures | 754.4 |
| Kitchen | 674.8 |
| Personal Care | 580.1 |
| Products at Public Facilities | 935.7 |
| Sports and Recreation | 749.1 |
| Toys & Children | 620.7 |
| Yard & Garden | 743.7 |

**Overall average text length:** 687.9

## Example Incidents (1 per severity)

**Severity 1** (Electronics):

> Tested the outlets on a wall surge protector/power strip and half of the outlets have their hot and neutral lines reversed. Other Amazon reviews confirm the same thing, so it is a faulty and dangerous design. Amazon (and other retailers) continue to sell the product....

**Severity 2** (Kitchen):

> The consumer would like to file a report on her unit, which she feels poses a fire hazard.



On 11/23/14 the consumer was in the kitchen washing dishes when she noticed that the unit turned on spontaneously. The unit began to run as if it was set to heat up an item. The consumer did not notice what was displayed on the control panel. The consumer was shocked that the unit turned on by itself and ...

**Severity 3** (Kitchen):

> The consumer says the front glass has a "catch" at the top but not at the bottom.  There is a double sided tape that holds the glass together.  The tape came loose, and the glass panel fell completely out.  The consumer says that the neighbor went to pick up the glass.  Cut her when she picked up glass panel.  Glass did not shatter.. The consumer contacted the manufacturer offered $60 to replace t...

**Severity 4** (Sports and Recreation):

> Consumer stated that her spouse was coasting down a paved hill on his bicycle approximately 30 miles per hour when the frame failed catastrophically above the pedal assembly, he was thrown forward over the handle bar as bike broke into 2 pieces to the concrete pavement on his head where he died instantly.  The victim was wearing a helmet during the incident. This happened in front of a Wayne fire ...

