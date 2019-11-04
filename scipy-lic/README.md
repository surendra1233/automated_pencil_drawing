```shell
ffmpeg -framerate 24 -i out/flow-%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p flow-field.mp4
```
