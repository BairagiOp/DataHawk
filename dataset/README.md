# DataHawk Benchmark Dataset

## Purpose
This directory contains the reproducible benchmark dataset for evaluating
DataHawk against baseline methods.

## Structure
```
dataset/
├── ground_truth/          # Ground truth JSON files (one per test case)
│   ├── ecom_001.json
│   ├── news_001.json
│   ├── jobs_001.json
│   └── ...
├── cached_pages/          # Cached HTML snapshots for reproducibility
│   ├── ecom_001.html
│   └── ...
└── README.md              # This file
```

## Format
Each ground truth file follows this structure:
```json
{
  "id": "ecom_001",
  "url": "https://example.com/product",
  "category": "ecommerce",
  "extraction_task": "Extract product name, price, rating",
  "schema": {
    "fields": {
      "product_name": {"type": "string", "required": true},
      "price": {"type": "number", "required": true}
    }
  },
  "ground_truth": [
    {"product_name": "Example Product", "price": 29.99}
  ],
  "cached_html_file": "cached_pages/ecom_001.html"
}
```

## Categories
- **ecommerce**: Product pages with prices, ratings, availability
- **news**: Articles with titles, authors, dates
- **jobs**: Job listings with titles, companies, salaries
- **docs**: Documentation/table pages
- **blog**: Blog posts with structured content

## Ethics
- All URLs are publicly accessible pages
- robots.txt is respected
- No authentication bypass
- No CAPTCHA circumvention
- Rate limits are respected
- Cached HTML is used for reproducible testing

## Adding New Entries
1. Create a ground truth JSON file in `ground_truth/`
2. Optionally save the page HTML in `cached_pages/`
3. Ensure the URL is publicly accessible
4. Verify ground truth accuracy manually
