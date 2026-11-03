fn main() {
    let endpoint = std::env::args().nth(1).expect("endpoint required");
    if endpoint == "smtp" {
        let host = std::env::args().nth(2).expect("SMTP host required");
        let bodies = layerfault::platform::weekly::NewsletterBodies {
            subject: "Controlled fixture".to_owned(),
            text: "Private sink fixture".to_owned(),
            markdown: "Private sink fixture".to_owned(),
            html: "<p>Private sink fixture</p>".to_owned(),
        };
        if let Err(error) = layerfault::platform::weekly::send_smtp(
            &bodies, "recipient@example.invalid", "sender@example.invalid", &host,
            Some("fixture-user"), Some("fixture-password"), false,
        ) {
            eprintln!("{error:#}");
            std::process::exit(1);
        }
        println!("Private SMTP sink delivery completed");
        return;
    }
    let input = layerfault::judge::JudgeInput {
        probe_id: "contract-fixture".to_owned(),
        category: "transport".to_owned(),
        prompt_excerpt: "benign synthetic fixture".to_owned(),
        response_excerpt: "benign synthetic response".to_owned(),
        base_response_excerpt: None,
        local_rule_ids: Vec::new(),
        local_classification: "INCONCLUSIVE".to_owned(),
    };
    match layerfault::judge::cloud_openai_compatible(&endpoint, "fixture", None, true, &input) {
        Ok(result) => println!("{result:?}"),
        Err(error) => {
            eprintln!("{error:#}");
            std::process::exit(1);
        }
    }
}
