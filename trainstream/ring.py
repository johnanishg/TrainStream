"""Ring all-reduce over TCP: how data-parallel ranks on different machines sum a gradient.

Each rank keeps two connections, one to the next rank and one from the
previous. A gradient is cut into as many segments as there are ranks; each
segment is summed as it travels once around the ring and the finished sums
travel around once more (csrc/ring.cpp). Whatever the number of ranks N, a
rank sends 2 (N - 1) / N of the gradient, sends and receives at the same time,
and ends with exactly the bits every other rank has.

Only the operating system's sockets are involved. The connections are made
here with the standard library and handed to the C++ side, which moves the
data without the interpreter lock. The ranks learn each other's addresses
through one small all_reduce on the torch.distributed group they already
share.

The address a rank announces is, in this order: TRAINSTREAM_RING_HOST; the
loopback address when every rank is on this machine; the address this machine
uses to reach MASTER_ADDR (the torchrun convention); this machine's host name.
"""

from __future__ import annotations

import os
import socket
import struct
from typing import List, Optional

import torch

from . import _C

_HELLO = 0x54535248  # "TSRH"
CONNECT_TIMEOUT = 60.0


def _tune(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, 4 << 20)
        except OSError:
            pass
    sock.setblocking(True)  # the C++ side sets its own timeouts on the OS socket


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    out = b""
    while len(out) < n:
        part = sock.recv(n - len(out))
        if not part:
            raise OSError("connection closed during the ring handshake")
        out += part
    return out


def local_address(single_node: bool) -> str:
    """The IPv4 address the other ranks reach this one at (see the module docstring)."""
    host = os.environ.get("TRAINSTREAM_RING_HOST")
    if host:
        return socket.gethostbyname(host)
    if single_node:
        return "127.0.0.1"
    master = os.environ.get("MASTER_ADDR")
    if master:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect((master, int(os.environ.get("MASTER_PORT", "29500"))))  # no packet is sent
            addr = probe.getsockname()[0]
            if not addr.startswith("127."):
                return addr
        except OSError:
            pass
        finally:
            probe.close()
    return socket.gethostbyname(socket.gethostname())


class RingAllReduce:
    """One rank's place in the ring: call it with a float32 host tensor to sum it over the ranks
    in place. Every rank must reduce the same tensors in the same order."""

    def __init__(self, to_next: socket.socket, from_prev: socket.socket, rank: int, world: int, timeout: int = 600):
        _tune(to_next)
        _tune(from_prev)
        self.rank, self.world = rank, world
        self._ring = _C.Ring(to_next.detach(), from_prev.detach(), rank, world, int(timeout))

    def __call__(self, buf: torch.Tensor) -> None:
        self._ring.allreduce(buf)

    def stats(self) -> dict:
        return dict(self._ring.stats())

    @classmethod
    def connect(cls, group, world: int, rank: int, timeout: int = 600) -> Optional["RingAllReduce"]:
        """Collective over the torch.distributed `group`: every rank listens, the addresses go
        around in one all_reduce, each rank connects to the next and accepts the previous.
        Returns None on every rank when any rank could not make its connections."""
        import torch.distributed as dist
        single = int(os.environ.get("LOCAL_WORLD_SIZE", world)) == world
        listener = to_next = from_prev = None
        ok = True
        try:
            host = local_address(single)
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1" if host.startswith("127.") else "0.0.0.0", 0))
            listener.listen(2)
            port = listener.getsockname()[1]
            mine = list(socket.inet_aton(host)) + [port]
        except OSError:
            ok, mine = False, [0, 0, 0, 0, 0]
        table = torch.zeros(world, 6, dtype=torch.int64)
        table[rank, :5] = torch.tensor(mine)
        table[rank, 5] = int(ok)
        dist.all_reduce(table, group=group)  # each row was written by one rank
        if int(table[:, 5].sum()) == world:
            try:
                nxt = table[(rank + 1) % world]
                to_next = socket.create_connection((socket.inet_ntoa(bytes(int(b) for b in nxt[:4])), int(nxt[4])),
                                                   timeout=CONNECT_TIMEOUT)
                to_next.sendall(struct.pack("<II", _HELLO, rank))
                listener.settimeout(CONNECT_TIMEOUT)
                from_prev, _ = listener.accept()
                from_prev.settimeout(CONNECT_TIMEOUT)
                magic, who = struct.unpack("<II", _recv_exact(from_prev, 8))
                if magic != _HELLO or who != (rank - 1) % world:
                    raise OSError("unexpected peer on the ring's listening port")
            except OSError:
                ok = False
        else:
            ok = False
        agreed = torch.tensor([int(ok)], dtype=torch.int64)
        dist.all_reduce(agreed, group=group)
        if listener is not None:
            listener.close()
        if int(agreed) != world:
            for s in (to_next, from_prev):
                if s is not None:
                    s.close()
            return None
        return cls(to_next, from_prev, rank, world, timeout)


def connect_local(world: int, timeout: int = 600) -> List[RingAllReduce]:
    """`world` connected ring members inside this process, over the loopback interface (tests,
    and trying the ring without launching several processes): call them from `world` threads."""
    listeners = []
    for _ in range(world):
        ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ls.bind(("127.0.0.1", 0))
        ls.listen(2)
        listeners.append(ls)
    to_next, from_prev = [None] * world, [None] * world
    for r in range(world):
        nxt = (r + 1) % world
        to_next[r] = socket.create_connection(("127.0.0.1", listeners[nxt].getsockname()[1]), timeout=CONNECT_TIMEOUT)
        from_prev[nxt], _ = listeners[nxt].accept()
    for ls in listeners:
        ls.close()
    return [RingAllReduce(to_next[r], from_prev[r], r, world, timeout) for r in range(world)]
