- The model we will use for music generation is called ACE-Step. The github repo can be found [here](https://github.com/ace-step/ACE-Step). The app diagram shows more details, check out `excalidraw.excalidraw` file in the root of the project
- In `Modal` we can create Volumes that can be used to store files like model weights and database dumps. We can share these volumes with different images (so we avoid downloading files to each image)

#### main.py notes

- The `MusicGenServer` class is the main class where we have the endpoints that will be consumed in the frontend
- The `load_models` method will be called when the container is cold and will load the models to the cache
