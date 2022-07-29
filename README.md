# film_edit_detection

### Synthetic Audio-Visual Dataset

Construction of synthetic Audio-Visual dataset. Our module generates 
artificial cuts (e.g. hard, dissolve, fade in/out) using both visual and audio data. 

```
cd data/syn_data
python3 audio_visual_synthesis.py [-h] --video_path VIDEO_PATH --annotation_path ANNOTATION_PATH --output_path OUTPUT_PATH --num_frames NUM_FRAMES --N N
```