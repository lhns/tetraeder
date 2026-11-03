#!/usr/bin/env bash
# Stops Track C's box jobs (session wrapper, detector campaign and its harness and sampler); patterns live here, not
# inline in ssh. The [x] brackets keep pgrep from matching this script's own command line.
pkill -f 'c_sessio[n].sh'
pkill -f 'detector_campaig[n].sh'
pkill -f 'detector_sampl[e].py'
pkill -f 'exit_regim[e]'
pkill -f 'cargo tes[t] --release -p tetraeder-model'
sleep 2
pgrep -fa 'detector_campaig[n]|c_sessio[n]' || echo stopped
