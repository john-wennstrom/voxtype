use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Request {
    pub previous: String,
    pub text: String,
    pub spans: Vec<Options>,
    pub timeout_ms: u64,
}

#[derive(Serialize, Deserialize)]
pub struct Options {
    pub original: String,
    pub start: usize,
    pub end: usize,
    pub terms: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub span: usize,
    pub option: usize,
}

impl Request {
    pub fn grammar(&self) -> String {
        let mut rules = Vec::new();
        for (position, span) in self.spans.iter().enumerate() {
            let options = (0..=span.terms.len())
                .map(|option| format!("\"{option}\""))
                .collect::<Vec<_>>()
                .join(" | ");
            rules.push(format!("span{position} ::= \"{{\\\"span\\\":{position},\\\"option\\\":\" ({options}) \"}}\""));
        }
        let sequence = (0..self.spans.len())
            .map(|position| format!("span{position}"))
            .collect::<Vec<_>>()
            .join(" \",\" ");
        format!("root ::= \"[\" {sequence} \"]\"\n{}", rules.join("\n"))
    }

    pub fn validate(&self) -> bool {
        !self.spans.is_empty()
            && self.spans.len() <= 64
            && (1..=10_000).contains(&self.timeout_ms)
            && self.text.len() + self.previous.len() <= 65_536
            && self.spans.iter().all(|span| {
                span.start < span.end
                    && !span.terms.is_empty()
                    && span.terms.len() <= 8
                    && span.original.len() <= 4096
                    && span
                        .terms
                        .iter()
                        .all(|term| !term.is_empty() && term.len() <= 4096)
            })
    }

    pub fn decisions(&self, bytes: &[u8]) -> Option<Vec<Decision>> {
        let decisions: Vec<Decision> = serde_json::from_slice(bytes).ok()?;
        if decisions.len() != self.spans.len()
            || decisions.iter().enumerate().any(|(position, decision)| {
                decision.span != position || decision.option > self.spans[position].terms.len()
            })
        {
            return None;
        }
        Some(decisions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_and_output_are_limited_to_supplied_options() {
        let request = Request {
            previous: String::new(),
            text: "get".into(),
            spans: vec![Options {
                original: "get".into(),
                start: 0,
                end: 1,
                terms: vec!["git".into()],
            }],
            timeout_ms: 300,
        };
        assert!(request.validate());
        assert!(request.grammar().contains("(\"0\" | \"1\")"));
        assert!(request.decisions(br#"[{"span":0,"option":1}]"#).is_some());
        for bytes in [
            br#"[{"span":0,"option":2}]"#.as_slice(),
            br#"[{"span":1,"option":1}]"#,
            br#"[]"#,
            br#"[{"span":0,"option":1,"text":"invented"}]"#,
        ] {
            assert!(request.decisions(bytes).is_none());
        }
    }
}
