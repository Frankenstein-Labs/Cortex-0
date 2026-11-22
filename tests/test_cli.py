from __future__ import annotations

from cortex.cli import build_parser, main


def test_parser_exposes_core_commands():
    parser = build_parser()
    args = parser.parse_args(["generate", "--prompt", "Bonjour", "--max-new-tokens", "4"])
    assert args.command == "generate"
    assert args.prompt == "Bonjour"


def test_init_db_command(tmp_path, capsys):
    code = main(["init-db", "--db-path", str(tmp_path / "db.sqlite")])
    assert code == 0
    output = capsys.readouterr().out
    assert "memories" in output
    assert (tmp_path / "db.sqlite").exists()


def test_publish_hf_without_token_is_blocked(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    code = main(["publish-hf", "--folder", str(folder)])
    assert code == 2
    assert "HF_TOKEN" in capsys.readouterr().out
