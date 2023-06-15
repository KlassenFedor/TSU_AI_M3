# TSU_AI_M3

### Description
Recognition of keywords in an audio stream using neural networks.

This application is able to process the audio data stream and extract keywords ("STONES") from it. You can train a model for your keywords using your own dataset. For this work, the dataset was formed by recording background sound and a set of one-second recordings of the keyword. Various changes were applied to the data to obtain more diversity.

The work was performed by the 3rd year student of Tomsk State University - Klassen Fedor

### Dataset structure:
- dataset:
  - background_clips
  - stones
    final_clips
  
    datset_json.json
    
    background_audio.wav

### How to use
Train model:
```
>python model.py train number_of_epochs
```

Work with audio stream:
```
>python stream_analyzer.py analyze_stream "your link"
```

Prepare dataset:
```
>python dataset_preparation.py prepare_dataset
```

### Sources
https://www.tensorflow.org/tutorials/audio/simple_audio?hl=en


https://github.com/jiaaro/pydub/blob/master/API.markdown


https://www.dlology.com/blog/how-to-do-real-time-trigger-word-detection-with-keras/#disqus_thread
