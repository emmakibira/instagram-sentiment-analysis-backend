#!/usr/bin/env python
"""Create an Instagram session file for use by the sentiment analyzer.

Instagram increasingly requires a login to read post metadata and comments
from certain networks. This script logs in once and saves an Instaloader
session file that the analyzer can load later.

Usage:
    python create_session.py --username myuser
    python create_session.py --username myuser --password 'hunter2'
    python create_session.py --username myuser --session session-myuser

Then set INSTAGRAM_SESSION_FILE=session-myuser in .env, or enter the session
path in the app sidebar (Settings -> Instagram credentials). A session file
is safer than storing your password in plaintext.

Two-factor authentication is handled interactively when enabled.
"""

from __future__ import annotations

import argparse
import getpass
import sys

import instaloader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Instagram username")
    parser.add_argument("--password", default=None, help="Instagram password (prompted if omitted)")
    parser.add_argument(
        "--session",
        default=None,
        help="Session file path (default: session-<username> in the current directory)",
    )
    args = parser.parse_args()

    password = args.password or getpass.getpass(f"Password for {args.username}: ")
    session_file = args.session or f"session-{args.username}"

    loader = instaloader.Instaloader()

    try:
        print(f"Logging in as {args.username} ...")
        loader.login(args.username, password)
    except instaloader.TwoFactorAuthRequiredException:
        code = input("Enter your two-factor authentication code: ").strip()
        loader.two_factor_login(code)
    except instaloader.BadCredentialsException:
        print("Error: invalid username or password.", file=sys.stderr)
        return 1
    except instaloader.ConnectionException as exc:
        print(f"Error: network problem while logging in: {exc}", file=sys.stderr)
        return 1

    loader.save_session_to_file(session_file)
    print(f"✓ Session saved to {session_file}")
    print(f"\nNext: add `INSTAGRAM_SESSION_FILE={session_file}` to your .env file, ")
    print("or paste that path in the app sidebar under Settings -> Instagram credentials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
