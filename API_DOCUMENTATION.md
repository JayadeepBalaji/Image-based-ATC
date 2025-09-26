# Animal Type Classification (ATC) - API Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Health Check
**GET** `/health`

Check the API health status and component availability.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-09-23T10:30:00",
  "components": {
    "database": true,
    "classifier": true,
    "image_processor": true
  }
}
```

### 2. Root Information
**GET** `/`

Get API information and available endpoints.

**Response:**
```json
{
  "message": "Animal Type Classification API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "health": "/health",
    "upload": "/upload",
    "history": "/history",
    "stats": "/stats"
  }
}
```

### 3. Upload and Classify Animal
**POST** `/upload`

Upload an animal image for classification.

**Parameters:**
- `file` (file): Image file (jpg, png, jpeg)
- `species` (form): Animal species ('cow' or 'buffalo')
- `animal_name` (form, optional): Name for the animal

**Example Request:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@cow_image.jpg" \
  -F "species=cow" \
  -F "animal_name=Bella"
```

**Response:**
```json
{
  "id": "uuid-string",
  "timestamp": "2024-09-23T10:30:00",
  "animal_name": "Bella",
  "species": "cow",
  "original_filename": "cow_image.jpg",
  "features": {
    "body_length_ratio": 1.42,
    "chest_depth_ratio": 0.73,
    "leg_length_ratio": 0.51,
    "body_condition_score": 0.82,
    "structural_soundness": 0.88,
    "udder_placement_score": 0.76,
    "productivity_indicators": {
      "body_proportions": 0.85,
      "chest_capacity": 0.79,
      "leg_structure": 0.83,
      "overall_conformation": 0.86
    },
    "measurements": {
      "body_length": 645,
      "body_height": 453,
      "body_area": 186532,
      "aspect_ratio": 1.42,
      "chest_width": 331
    }
  },
  "classification": {
    "productivity_class": "High",
    "confidence": 0.87,
    "probability_distribution": {
      "High": 0.87,
      "Medium": 0.11,
      "Low": 0.02
    },
    "overall_score": 84.2,
    "trait_scores": {
      "body_length": {
        "value": 1.42,
        "score": 99.4,
        "ideal_range": "1.3 - 1.5",
        "description": "Optimal body length indicates good growth potential"
      },
      "chest_depth": {
        "value": 0.73,
        "score": 94.9,
        "ideal_range": "0.65 - 0.80",
        "description": "Good chest depth indicates lung and heart capacity"
      }
    },
    "recommendations": [
      "Excellent animal with high productivity potential",
      "Maintain good nutrition and regular health checks",
      "Consider for breeding programs"
    ]
  },
  "success": true
}
```

### 4. Get Classification History
**GET** `/history?limit=10`

Retrieve recent classification history.

**Parameters:**
- `limit` (query, optional): Maximum number of records (default: 10)

**Response:**
```json
{
  "history": [
    {
      "id": "uuid-string",
      "timestamp": "2024-09-23T10:30:00",
      "animal_name": "Bella",
      "species": "cow",
      "classification": {
        "productivity_class": "High",
        "confidence": 0.87,
        "overall_score": 84.2
      }
    }
  ],
  "count": 1,
  "limit": 10
}
```

### 5. Get Statistics
**GET** `/stats`

Get system usage statistics.

**Response:**
```json
{
  "total_classifications": 45,
  "high_productivity_count": 15,
  "medium_productivity_count": 22,
  "low_productivity_count": 8,
  "average_score": 67.3,
  "average_confidence": 0.78,
  "high_productivity_percentage": 33.3,
  "medium_productivity_percentage": 48.9,
  "low_productivity_percentage": 17.8,
  "species_breakdown": {
    "cow": 28,
    "buffalo": 17
  },
  "recent_activity": [
    {"date": "2024-09-23", "count": 5},
    {"date": "2024-09-22", "count": 3}
  ]
}
```

### 6. Delete Classification
**DELETE** `/history/{classification_id}`

Delete a specific classification record.

**Parameters:**
- `classification_id` (path): ID of the classification to delete

**Response:**
```json
{
  "message": "Classification deleted successfully"
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- `400`: Bad Request (invalid input)
- `404`: Not Found (resource doesn't exist)
- `500`: Internal Server Error (server-side error)

## Feature Extraction Details

The system extracts the following features from animal images:

### Primary Measurements
- **Body Length Ratio**: Aspect ratio of the animal (length/height)
- **Chest Depth Ratio**: Chest width relative to body height
- **Leg Length Ratio**: Leg length relative to body height
- **Body Condition Score**: Overall body condition assessment
- **Structural Soundness**: Structural integrity score
- **Udder Placement Score**: Estimated udder quality (for females)

### Productivity Indicators
- **Body Proportions**: Overall body shape assessment
- **Chest Capacity**: Breathing and feed intake capacity
- **Leg Structure**: Mobility and longevity indicators
- **Overall Conformation**: General structural quality

### Raw Measurements
- **Body Length**: Pixel measurements of body length
- **Body Height**: Pixel measurements of body height
- **Body Area**: Total body area in pixels
- **Chest Width**: Chest measurement in pixels
- **Leg Span**: Distance between front and back legs

## Classification Classes

### High Productivity
- Animals with optimal body structure for production
- High scores across all trait categories
- Excellent breeding and production potential

### Medium Productivity
- Animals with good structure and moderate limitations
- Average to good scores with some areas for improvement
- Suitable for production with proper management

### Low Productivity
- Animals with structural issues affecting productivity
- Below-average scores in multiple categories
- May require special management or may not be suitable for breeding

## Image Requirements

For best results, uploaded images should meet these criteria:

### ✅ Good Images
- Clear side view of the animal
- Good lighting conditions
- Minimal background clutter
- Full body visible
- Animal standing upright
- High resolution (recommended: 800x600 or higher)

### ❌ Avoid
- Blurry or low-quality images
- Extreme angles (front/back view only)
- Poor lighting or shadows
- Partially obscured animals
- Animals lying down
- Very low resolution images

## Rate Limits

Currently, no rate limits are enforced, but for production use consider:
- Maximum 100 requests per minute per IP
- Maximum file size: 10MB
- Supported formats: JPG, PNG, JPEG