"""Tests for the SMTP RCPT TO check in forge.importers.smtp_verifier.

A one-connection fake mail server on localhost stands in for the MX host, so the real
smtplib and ssl code paths run without touching the network.
"""

import importlib
import socket
import threading

import pytest

from forge.importers import smtp_verifier


class _FakeMX:
    """Localhost SMTP server for one connection; records every command the client sends."""

    def __init__(self, starttls: bool, rcpt_code: int = 250):
        self.starttls = starttls
        self.rcpt_code = rcpt_code
        self.commands: list = []
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        conn, _ = self._sock.accept()
        with conn, self._sock:
            f = conn.makefile("rwb")

            def send(line):
                f.write(line.encode() + b"\r\n")
                f.flush()

            send("220 fake.mx ESMTP")
            while True:
                raw = f.readline()
                if not raw:
                    return
                cmd = raw.decode(errors="replace").strip().upper()
                self.commands.append(cmd)
                verb = cmd.split(" ", 1)[0]
                if verb == "EHLO":
                    if self.starttls:
                        send("250-fake.mx")
                        send("250 STARTTLS")
                    else:
                        send("250 fake.mx")
                elif verb == "STARTTLS":
                    # No TLS here: hang up, so the client's handshake fails.
                    send("220 go ahead")
                    return
                elif verb == "RCPT":
                    send(f"{self.rcpt_code} result")
                elif verb == "QUIT":
                    send("221 bye")
                    return
                else:
                    send("250 OK")

    def join(self):
        self._thread.join(timeout=5)


@pytest.fixture
def route_to(monkeypatch):
    """Send smtplib's outbound connection to a _FakeMX instead of port 25."""
    real = socket.create_connection

    def _route(mx):
        def fake(address, *args, **kwargs):
            return real(("127.0.0.1", mx.port), *args, **kwargs)

        monkeypatch.setattr(socket, "create_connection", fake)

    return _route


class TestSMTPCheck:
    def test_starttls_server_does_not_crash_the_check(self, route_to):
        mx = _FakeMX(starttls=True)
        route_to(mx)
        assert smtp_verifier._smtp_check("info@shop.test", "mx.shop.test") is None
        mx.join()
        assert "STARTTLS" in mx.commands

    def test_returns_the_rcpt_code(self, route_to, monkeypatch):
        monkeypatch.setattr(smtp_verifier, "PROBE_FROM", "")
        mx = _FakeMX(starttls=False, rcpt_code=550)
        route_to(mx)
        assert smtp_verifier._smtp_check("nobody@shop.test", "mx.shop.test") == 550
        mx.join()
        assert "MAIL FROM:<>" in mx.commands
        assert "RCPT TO:<NOBODY@SHOP.TEST>" in mx.commands


class TestProbeSender:
    def test_default_sender_is_the_null_sender(self, monkeypatch):
        monkeypatch.delenv("FORGE_SMTP_FROM", raising=False)
        assert importlib.reload(smtp_verifier).PROBE_FROM == ""

    def test_sender_from_env(self, monkeypatch):
        monkeypatch.setenv("FORGE_SMTP_FROM", " verify@mydomain.com ")
        assert importlib.reload(smtp_verifier).PROBE_FROM == "verify@mydomain.com"
        monkeypatch.delenv("FORGE_SMTP_FROM")
        importlib.reload(smtp_verifier)
