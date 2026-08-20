from io import BytesIO
from runpy import run_path


class Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def test_public_data_download_is_bounded_and_writes_provenance(tmp_path) -> None:
    module = run_path("scripts/prepare_public_data.py")
    payload = ("verified public text " * 100).encode()
    module["download"].__globals__["urlopen"] = lambda *_args, **_kwargs: Response(payload)
    output = tmp_path / "sample.txt"
    result = module["download"]("tinyshakespeare", 1_024, output)
    assert output.exists()
    assert output.stat().st_size <= 1_024
    assert output.with_suffix(".meta.json").exists()
    assert result["source"] == "tinyshakespeare"
    assert len(result["sha256"]) == 64
