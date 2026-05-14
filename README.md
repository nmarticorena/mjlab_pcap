# Demo of PCAP sim using mjlab

## Install
### Pixi
This repo use `pixi` to manage all dependencies and environments, it can be installed by simply running

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

More info in [pixi docs](https://pixi.prefix.dev/latest/#installation).

### Deps
All dependencies are stated in [pixi.toml](pixi.toml), so to install all deps will be installed when running the demo script
```bash
pixi run demo
```


## Demo
At the moment this repo is a *extremely* simple env, in which we load a panda arm and a sampled tree from [pcap](https://sites.google.com/view/pcap/home#h.hzsqdc1jjw14) using its parametric L-system approach.


This command will install `python`, `mjlab` including the multiple deps such as `warp`, `mjviser` and [others](https://github.com/mujocolab/mjlab/blob/a0ba05890a2ea4111b33c9cbb85f690bf19ca434/pyproject.toml#L33-L51). 
Once all deps are installed it will compile the warp kernels and then lunch a prompt with the viewer to visualize the scene.
