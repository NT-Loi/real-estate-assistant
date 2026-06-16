# Real Estate Query Test Set

This fixture contains 500 Vietnamese natural-language test cases for the real-estate assistant.

## Files

- `real_estate_queries.testset.jsonl`: JSONL dataset, one test case per line.
- `validate_testset.py`: lightweight validator for count, schema, category balance, duplicate IDs, and JSONL validity.

## Categories

- `property_filter_search`: 167 cases for intent detection and extraction of listing/search filters such as `listing_type`, `property_type`, `location`, `district`, `price_vnd`, `area_m2`, `bedrooms`, `bathrooms`, `legal`, `furniture`, direction, nearby distance, and buy/rent purpose.
- `nearby_amenities`: 167 cases for extracting amenity category, radius or travel time, property reference, coordinates/address, and comparison requests. The dataset aligns with the codebase's OSM categories: `school`, `hospital`, `transit_station`, `park`, `shopping_mall`, and `supermarket`; some cases intentionally ask for unsupported or more specific amenities to test graceful handling.
- `financial_calculation`: 166 cases for extracting financial inputs and checking expected formulas for mortgage payment, loan amount, total interest, rental yield, ROI, cash flow, initial costs, affordability, and rent break-even.

## How To Use

Use the dataset to evaluate:

- intent detection: map each query to the expected intent;
- entity extraction: compare extracted values to `required_filters_or_inputs`;
- search filter mapping: convert Vietnamese phrases into listing filters used by the app;
- amenities lookup: map amenity words and distances to nearby POI requests;
- financial calculation: validate extracted numeric inputs and formula selection.

The cases cover normal queries, missing information, ambiguous phrasing, invalid values, multi-constraint searches, comparisons, boundary values, short queries, long queries, mild typos, synonym usage, negative constraints, and mixed units such as ty, trieu, m2, km, and minutes.

## Validation

Run from the repository root:

```bash
python tests/fixtures/validate_testset.py
```

The script expects exactly 500 cases, near-even category counts of 167/167/166, unique IDs, required fields on every case, valid JSONL, and non-empty `required_filters_or_inputs` for `normal` cases.
