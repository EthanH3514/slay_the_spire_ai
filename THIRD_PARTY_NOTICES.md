# Third-party notices

Spire AI is an unofficial project for Slay the Spire. The headless runner uses
the [maintained sts_lightspeed fork](https://github.com/EthanH3514/sts_lightspeed)
at the revision recorded in `src/spire_ai/simulator-lock.json`. The build
retrieves the simulator and its nlohmann/json and pybind11 dependencies. Their
license files are included in the respective upstream checkouts.

The bundled rule catalogue contains simulator-derived metadata. Simulator
copyright and permission notice follows. The repository distributes no
commercial game executable or art, audio, mod binaries or training dataset.
The model manifest identifies the individual inference exports by digest.
The upstream run-policy work used the public
[Slay the Spire metrics dump](https://www.reddit.com/r/slaythespire/comments/jt5y1w/77_million_runs_an_sts_metrics_dump/)
for offline initialization. This repository includes the resulting inference
weights.

## sts_lightspeed

MIT License

Copyright (c) 2021 gamerpuppy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
