## Overview

CausalLab is an Interactive Causal Analysis Tool.


## Installation

The latest [YLearn](https://github.com/DataCanvasIO/YLearn) is required to run CausalLab, so install it from the latest source code before installing CausalLab:

```console
pip install "torch<2.0.0" "pyro-ppl<1.8.5" gcastle
pip install git+https://github.com/DataCanvasIO/YLearn.git
```

Now, one can install CausalLab from the source:

```console
git clone https://github.com/DataCanvasIO/CausalLab
cd CausalLab
pip install .
```
 
## Startup

Run `causal_lab` to startup CausalLab http server on localhost with default port(5006):

```console
causal_lab
```


To accept request from other computers, specify local `host_ip` and `port` to startup CausalLab http server:

```console
causal_lab --address <host_ip> --port <port> --allow-websocket-origin=<host_ip>:<port>
```

eg:

```console
causal_lab --address 172.20.51.203 --port 15006 --allow-websocket-origin=172.20.51.203:15006
```


## License
See the [LICENSE](LICENSE) file for license rights and limitations (Apache-2.0).
