# CPAEP — GeMM/GeMV Accelerator

A 512-multiplier accelerator for GeMM and GeMV with signed int8 inputs and int32 outputs.

## Run the simulation

Run the commands from the repository root in a Linux terminal with QuestaSim available. Enable it if needed:

```bash
source questa_setup.sh
```

Run the complete accelerator testbench without the GUI:

```bash
make TEST_MODULE=tb_mac_array_gemm questasim-run
```

Or with the GUI:

```bash
make TEST_MODULE=tb_mac_array_gemm questasim-run-gui
```

To run another testbench, replace `tb_mac_array_gemm` in the commands above with the corresponding name:

| Testbench | What it tests |
| --- | --- |
| `tb_mac_array_gemm` | Complete accelerator: GeMM/GeMV correctness, boundary cases, reset/restart behavior and SRAM protocol checks. |
| `tb_mac_array` | Array dot products, accumulation across blocks and contexts, accumulator clearing and reset. |
| `tb_gemm_controller_mac_array` | Controller indices, pipeline timing, stalls, dimension boundaries and consecutive jobs without reset. |
| `tb_one_mac_gemm` | Original single-MAC accelerator on six small GeMM/GeMV cases. |
| `tb_mac_pe` | Random signed dot products in the supplied MAC PE. |
| `tb_ceiling_counter` | Tick-driven counting with different ceiling values and clearing between tests. |
| `tb_single_port_memory` | Sequential writes and reads of random data, compared against reference values. |

In the global `tb_mac_array_gemm` testbench, the SRAM negative test intentionally prints eight error messages. The simulation must still end with the message that all matrix cases and reset/restart tests passed.

## Generate the roofline plot

After a successful simulation exports `roofline_results.csv`, run:

```bash
python3 -m pip install matplotlib  # Only needed once
python3 plot_roofline.py roofline_results.csv
```

The figures are saved in `roofline/`.
