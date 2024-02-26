# Dataset Structure
After running the file `generate_dataset.py`, the structure of the `avlmaps_data_dir` will look like the following. We will walk you through the data step by step.
```text
avlmaps_data_dir
    ├── 5LpN3gDmAk7_1
    │   ├── poses.txt
    │   ├── audio_video
    │   │   ├── 000000
    │   │   │   ├── meta.txt
    │   │   │   ├── poses.txt
    │   │   │   ├── output.mp4
    │   │   │   ├── output_level_1.wav
    │   │   │   ├── output_level_2.wav
    │   │   │   ├── output_level_3.wav
    │   │   │   ├── output_with_audio_level_1.mp4
    │   │   │   ├── output_with_audio_level_2.mp4
    │   │   │   ├── output_with_audio_level_3.mp4
    │   │   │   ├── range_and_audio_meta_level_1.txt
    │   │   │   ├── range_and_audio_meta_level_2.txt
    │   │   │   ├── range_and_audio_meta_level_3.txt
    │   │   │   ├── rgb
    │   │   │   |   ├── 000000.png
    │   │   │   |   ├── ...
    │   │   ├── 000001
    │   │   ├── ...
    │   ├── depth
    │   │   ├── 000000.npy
    │   │   ├── ...
    │   ├── rgb
    │   │   ├── 000000.png
    │   │   ├── ...
    │   ├── semantic
    │   │   ├── 000000.npy
    │   │   ├── ...
    ├── gTV8FGcVJC9_1
    │   ├── ...
    ├── jh4fc5c5qoQ_1
    │   ├── ...
    ...
```

1. Under the `avlmaps_data_dir` folder, there are scene folders named after the scene name in Matterport3D (like `5LpN3gDmAk7`, `gTV8FGcVJC9`, `jh4fc5c5qoQ` etc.) with a suffix number `_%d` indicating different floors in the scene. 
    <details>
    <summary>The 10 scenes are listed here</summary>
    <pre><code>
        avlmaps_data_dir
        ├── 5LpN3gDmAk7_1
        ├── gTV8FGcVJC9_1
        ├── jh4fc5c5qoQ_1
        ├── JmbYfDe2QKZ_1
        ├── JmbYfDe2QKZ_2
        ├── mJXqzFtmKg4_1
        ├── ur6pFq6Qu1A_1
        ├── UwV83HsGsw3_1
        ├── Vt2qJdWjCF2_1
        └── YmJkqBEsHnH_1
    </code></pre>

    </details>

2. Under each scene folder (take `5LpN3gDmAk7_1` as an example), there are `audio_video`, `rgb`, `depth`, and `semantic` folders and a `poses.txt` files.

    <details>
    <summary>The overview of the contents under a scene folder</summary>
    <pre><code>
        5LpN3gDmAk7_1
        ├── audio_video
        ├── rgb
        ├── depth
        ├── semantic
        └── poses.txt
    </code></pre>

    </details>
    
    * In `rgb` folder, there are all the RGB images collected in the scene. All files are in `.png` format.
    * In `depth` folder, there are all the depth images collected in the scene. All files are in `.npy` format, each of which contains a `(W, H, 1)` `np.ndarray`. Each value in the array indicates the depth value in meter.
    * In `semantic` folder, there are all the GT object-id labeled pixels for all RGB images. All files are in `.npy` format. To convert such object-id labels to semantic labels, we need to load the habitat Simulator and access the semantic information from the scene. This folder is not required for the AVLMaps mapping. But it can be used for creating a GT semantic map. You can delete the folder for the reduction of disk space occupation.
        ```python
            import habitat_sim
            import numpy as np
            # create a cfg as in the generate_scene_data() function in the file generate_dataset.py
            sim = habitat_sim.Simulator(cfg)
            scene = sim.semantic_scene
            objs = scene.objects
            obj2cls = {int(obj.id.split("_")[-1]): (obj.category.index(), obj.category.name()) for obj in scene.objects}

            semantic = np.load("avlmaps_data_dir/5LpN3gDmAk7_1/semantic/000000.npy")
            # for pixel (0, 0)
            pixel_category_id, pixel_category_name = obj2cls[semantic[0, 0]]

        ```
    * The `poses.txt` file save the position and quaternion of all frames. Each line in the file save the `x` `y` `z` `qx` `qy` `qz` `qw` values in this order.
    * The `audio_video` folder saved all the data for audio-based mapping. We will explain in detail in the following section.

3. Under the `audio_video` folder inside each scene folder, there are 20 subfolders containing the videos with inserted ESC50 audios.
    <details>
    <summary>The overview of the contents under an <code>audio_video</code> folder</summary>
    <pre><code>
        audio_video
        ├── 000000
        │   ├── meta.txt
        │   ├── poses.txt
        │   ├── output.mp4
        │   ├── output_level_1.wav
        │   ├── output_level_2.wav
        │   ├── output_level_3.wav
        │   ├── output_with_audio_level_1.mp4
        │   ├── output_with_audio_level_2.mp4
        │   ├── output_with_audio_level_3.mp4
        │   ├── range_and_audio_meta_level_1.txt
        │   ├── range_and_audio_meta_level_2.txt
        │   ├── range_and_audio_meta_level_3.txt
        │   ├── rgb
        │   |   ├── 000000.png
        │   |   ├── ...
        ├── 000001
        ├── ...
        └── 000019
    </code></pre>

    </details>

    In each of the subfolder (such as `000000`), there are following files:
    * `meta.txt`: the frame ranges where audios are inserted in the video.
    * `poses.txt`: the poses of all the video frames.
    * `output.mp4`: the video without audio insersion.
    * `output_level_<1,2,3>.wav`: the randomly selected audios at three different categories of considered sounds for audio insersion. The corresponding category for each level can be found at [config/sound_config/sound_config.yaml](../config/sound_config/sound_config.yaml).
    * `output_with_audio_level_<1,2,3>.mp4`: the videos with inserted audios. The video comes from `output.mp4` and the inserted audio for corresponding level is from `output_level_<1,2,3>.wav`.
    * `range_and_audio_meta_level_<1,2,3>.txt`: the meta data for the audio video. 

        <details>
        <summary> An example is shown below (avlmaps_data_dir/5LpN3gDmAk7_1/audio_video/000000/range_and_audio_meta_level_1.txt)</summary>
        <pre><code>
        0,125,0.0,5.0,clock tick,/home/huang/hcg/projects/vln/data/ESC-50-master/audio/1-62850-A-38.wav
        </code></pre>
            These values indicate: the audio starting frame id, the audio end frame id, the audio start time in the video (s), the audio end time the video (s), the GT semantics of the inserted sound, the path to the sound file in ESC-50 dataset.
        </details>

    * `rgb`: the RGB images of all frames in the video. This part is no longer required after generating the videos above. You can choose to manually delete them to reduce the required disk space.


## Inspect Audio Video Statistics

Run the following command to see the number of audio videos for each scene:

```bash
python dataset/dataset_statistics.py
```

We can see that there are 82 `level_1` sounds, 126 `level_2` sounds, and 127 `level_3` sounds inserted to the videos.

<details>
<summary>The output is shown here</summary>

```text
5LpN3gDmAk7_1
  level_1: 10
  level_2: 17
  level_3: 17
JmbYfDe2QKZ_1
  level_1: 9
  level_2: 9
  level_3: 9
JmbYfDe2QKZ_2
  level_1: 9
  level_2: 11
  level_3: 11
UwV83HsGsw3_1
  level_1: 8
  level_2: 13
  level_3: 13
Vt2qJdWjCF2_1
  level_1: 9
  level_2: 15
  level_3: 15
YmJkqBEsHnH_1
  level_1: 6
  level_2: 6
  level_3: 6
gTV8FGcVJC9_1
  level_1: 7
  level_2: 14
  level_3: 14
jh4fc5c5qoQ_1
  level_1: 7
  level_2: 7
  level_3: 7
mJXqzFtmKg4_1
  level_1: 9
  level_2: 17
  level_3: 17
ur6pFq6Qu1A_1
  level_1: 8
  level_2: 17
Overall
  level_1: 82
  level_2: 126
  level_3: 127
```

</details>
