from dialogue_locator.captions import discover_captions


def test_caption_metadata_normalization_and_languages() -> None:
    inventory = discover_captions(
        {
            "subtitles": {"en": [{
                "ext": "vtt",
                "url": "https://example.test/en.vtt",
                "name": "English",
                "protocol": "https",
                "impersonate": True,
                "http_headers": {"Referer": "https://example.test/"},
            }]},
            "automatic_captions": {"fr": [{"ext": "srv3"}], "en": [{"ext": "json3"}]},
        }
    )

    assert len(inventory.platform_subtitles) == 1
    assert len(inventory.automatic_captions) == 2
    assert inventory.available_languages == ["en", "fr"]
    assert inventory.platform_subtitles[0].extension == "vtt"
    assert inventory.platform_subtitles[0].protocol == "https"
    assert inventory.platform_subtitles[0].impersonate is True
    assert inventory.platform_subtitles[0].http_headers == {"Referer": "https://example.test/"}


def test_missing_or_malformed_caption_metadata_is_safe() -> None:
    inventory = discover_captions({"subtitles": None, "automatic_captions": ["unexpected"]})
    assert inventory.platform_subtitles == []
    assert inventory.automatic_captions == []
    assert inventory.available_languages == []
