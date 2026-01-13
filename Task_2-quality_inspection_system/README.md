# PCB Defect Detection System

automated visual inspection system for detecting defects on PCBs

## what it does

- finds 4 types of defects: scratches, missing components, solder bridges, discoloration
- gives bounding boxes and center (x,y) coordinates for each defect
- rates defect severity (low/medium/high/critical)
- calculates confidence scores for each detection
- gives overall quality score and pass/fail recommendation

## setup

```
pip install -r requirements.txt
```

## usage

### single image
```
python src/defect_detector.py path/to/image.png --save
```

### batch processing
```
python src/batch_processor.py images/defective output/results
```

### generate sample images
```
python src/generate_samples.py
```

## output format

the script outputs:
- annotated image with bounding boxes around defects
- json file with all detection details

json output looks like:
```json
{
  "image": "pcb_image.png",
  "has_defects": true,
  "defect_count": 3,
  "quality_score": 65.5,
  "recommendation": "REWORK - need to fix some stuff",
  "defects": [
    {
      "id": 1,
      "type": "scratch",
      "confidence": 0.85,
      "bbox": {"x": 100, "y": 200, "width": 50, "height": 20},
      "center_x": 125,
      "center_y": 210,
      "severity": "medium",
      "area": 1000
    }
  ]
}
```

## project structure

```
quality_inspection_system/
├── src/
│   ├── defect_detector.py     # main detection code
│   ├── batch_processor.py     # process multiple images
│   ├── generate_samples.py    # make test images
│   └── download_images.py     # download sample images
├── images/
│   ├── defective/             # images with defects
│   ├── non_defective/         # good images
│   └── annotated/             # annotation json files
├── output/                    # results go here
├── requirements.txt
└── README.md
```

## defect types detected

1. **scratch** - linear marks on PCB surface
2. **missing_component** - empty pads where parts should be
3. **solder_bridge** - unwanted solder connections between pads
4. **discoloration** - color changes from heat/oxidation damage

## severity levels

- low: small defect, probably ok
- medium: noticeable defect, should check
- high: significant defect, needs fixing
- critical: major defect, reject the board

## recommendations

- PASS: quality score >= 95
- REVIEW: score 80-95
- REWORK: score 60-80
- REJECT: score < 60
