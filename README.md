# film_edit_detection

### Synthetic Audio-Visual Dataset

Construction of synthetic Audio-Visual dataset. Our module generates 
artificial cuts (e.g. hard, dissolve, fade in/out) using both visual and audio data. 

```
pip install -r requirements.txt
cd data/syn_data
python3 audio_visual_synthesis.py [-h] video_path annotation_path output_path num_frames N
```
