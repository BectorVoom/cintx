pub fn canonicalize_symbol_name(raw: &str) -> String {
    canonicalize_token(raw, '_', false)
}

pub fn canonicalize_profile_label(raw: &str) -> String {
    let normalized = canonicalize_token(raw, '-', true);
    if normalized.is_empty() {
        return normalized;
    }
    let mut components = normalized
        .split('+')
        .filter(|part| !part.is_empty())
        .map(canonicalize_profile_component)
        .collect::<Vec<_>>();
    components.sort();
    components.dedup();
    if components.len() == 2
        && components.contains(&"with-f12".to_string())
        && components.contains(&"with-4c1e".to_string())
    {
        return "with-f12+with-4c1e".to_string();
    }
    components.join("+")
}

fn canonicalize_profile_component(component: &str) -> String {
    match component {
        "base" => "base".to_string(),
        "f12" | "with-f12" => "with-f12".to_string(),
        "4c1e" | "with-4c1e" => "with-4c1e".to_string(),
        other => other.to_string(),
    }
}

fn canonicalize_token(raw: &str, separator: char, keep_plus: bool) -> String {
    let mut canonical = String::new();
    let mut pending_separator = false;
    let mut prev_is_plus = false;

    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_separator && !canonical.is_empty() && !prev_is_plus {
                canonical.push(separator);
            }
            canonical.push(ch.to_ascii_lowercase());
            pending_separator = false;
            prev_is_plus = false;
            continue;
        }
        if keep_plus && ch == '+' {
            if !canonical.is_empty() && !prev_is_plus {
                canonical.push('+');
                prev_is_plus = true;
            }
            pending_separator = false;
            continue;
        }
        pending_separator = true;
    }

    canonical
        .trim_matches(separator)
        .trim_matches('+')
        .to_string()
}
