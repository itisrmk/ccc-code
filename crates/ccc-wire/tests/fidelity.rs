//! Checks round-trip fidelity on real captured traffic, which never enters the repo:
//! `CCC_FIDELITY_DIR=<ccc run --dump dir> cargo test -p ccc-wire --test fidelity -- --ignored`

use std::fs;

use ccc_wire::Request;

#[test]
#[ignore = "needs a local traffic dump named by CCC_FIDELITY_DIR"]
fn captured_requests_reserialize_byte_identically() {
    let dir = std::env::var("CCC_FIDELITY_DIR").expect("CCC_FIDELITY_DIR should be set");
    let mut checked = 0;
    for entry in fs::read_dir(&dir).expect("dump dir should be readable") {
        let path = entry.expect("dir entry should be readable").path();
        if !path.to_string_lossy().ends_with(".req-body.json") {
            continue;
        }
        let body = fs::read(&path).expect("dump file should be readable");
        let Ok(request) = Request::parse(&body) else {
            continue;
        };
        assert!(
            request.to_bytes() == body,
            "{} changed on round trip",
            path.display()
        );
        checked += 1;
    }
    assert!(checked > 0, "no request bodies found in {dir}");
    println!("{checked} captured requests are byte-identical after a round trip");
}
