# What “the plug catching against the port” means

It means **the plug touches part of the port housing instead of sliding cleanly into its opening**. It does not mean the plug blocks the entire port.

Imagine pushing a rectangular plug into a rectangular socket. If the plug is slightly sideways, its corner or outer casing can hit the socket's rim or inner wall. The robot keeps commanding forward movement, but the plug stops and the contact force rises.

```text
Clean insertion:                 Possible blocked insertion:

       plug                              plug
        ||                                ||
        vv                                vv
     |      |                         |      |
     |  ||  |                         |     || <- casing meets edge/wall
     |  vv  |                         |______|
      opening                          port housing
```

The drawing illustrates the mechanism; it is not a reconstruction of the exact contact surface.

In our reproduced failure:

- The simulator recorded contact between the **SC plug's outer housing and the selected port's housing**.
- The recorded plug-tip frame was about **1.58 mm sideways** from the port frame when the failure was reported.
- A contact-force spike occurred, and forward progress stopped.

These observations are consistent with the plug catching at the opening or against a port wall. **We have not isolated the exact rim, corner, or wall responsible.** The contact identities establish which parts touched; they do not alone prove the precise jamming mechanism. The original historical 3.5 N episode did not have these complete contact records, so its exact contact remains an inference from the reproduced failure.

Also, merely touching the port is normal during insertion. The failure is **contact that prevents further progress**, rather than contact by itself.

The failed retreat is a separate observation: an upward withdrawal was commanded, but the TCP barely moved. That does not yet prove that the plug was mechanically wedged; insufficient recovery control authority could also contribute.
