# Security

## Model

idsbench is a command line program that **reads CSV files and writes a report**. It opens no network connection, starts
no service and runs no other program. The files it reads are public research data, but they are still treated as
untrusted input.

- **Reading**: CSV files are parsed by pandas, text columns are only ever used as labels and category values, and a file
  that does not have the expected columns is refused with a message that lists what was found. Nothing is evaluated,
  imported or executed from a data file.
- **No pickles are read.** The benchmark trains models in memory, scores them and throws them away; no model file is
  saved or loaded, so there is no unpickling of untrusted data anywhere in the program.
- **Output**: the report is JSON, Markdown and a PNG. Text that comes from the data (attack names, column values) is
  written into Markdown tables as it is; if you publish a report made from your own data, read it first.
- **The Docker image** runs as an unprivileged user and works with a read-only root filesystem (plus `--tmpfs /tmp` for
  matplotlib's cache), no capabilities and `--network none`. Mount the datasets read-only (`-v ...:/datasets:ro`).
- **No data is stored here.** The datasets are not in the repository and `datasets/` is ignored by git.

What it does not do: it is a research benchmark, not a detector for production. A model that scores well here has
been tested on two old, lab-made datasets (see [limitations](docs/limitations.md)); do not deploy one because of this
table.

## Reporting

Please report vulnerabilities privately through GitHub's "Report a vulnerability" (Security tab), not in a public issue.
