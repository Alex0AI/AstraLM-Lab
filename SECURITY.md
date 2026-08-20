# Security policy

Do not post secrets or exploitable details in a public issue. Report a suspected
vulnerability privately through GitHub's security advisory interface. Include a
minimal reproduction, affected version, impact, and suggested mitigation when
possible.

AstraLM loads PyTorch checkpoints, which may contain pickled Python objects. Only
load checkpoints you trust. The project never treats an untrusted checkpoint as
safe input.

