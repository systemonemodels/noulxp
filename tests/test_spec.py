"""The standard's versions, under its name and under the one it had until 0.3.1."""

from noulxp.spec import BASE, STANDARD, at_least, canonical, parse_standard, same_version, supported


def test_versions_under_the_old_name_are_the_same_versions() -> None:
    assert supported("noulxp/0.1") and supported("noulxp/0.2")
    assert supported("odxp/0.1") and supported("odxp/0.2")
    assert not supported("noulxp/0.3") and not supported("odxp/0.3") and not supported("0.2")
    assert canonical("odxp/0.2") == "noulxp/0.2" == STANDARD
    assert parse_standard("odxp/0.1") == parse_standard(BASE) == (0, 1)
    assert at_least("odxp/0.2", "noulxp/0.2") and not at_least("odxp/0.1", "noulxp/0.2")
    assert same_version("odxp/0.2", "noulxp/0.2") and not same_version("odxp/0.1", "noulxp/0.2")
