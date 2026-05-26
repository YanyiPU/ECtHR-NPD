# Article 6 Limb Router

Article 6 must be routed to either:

- `art6_civil`
- `art6_criminal`

## Civil Indicators

- civil rights and obligations
- administrative, family, employment, pensions, enforcement, compensation,
  access to tribunal in non-criminal context

## Criminal Indicators

- criminal charge
- conviction, sentence, prosecutor, defence rights, witnesses, interpreter,
  presumption of innocence

## Ambiguous Cases

If the case text is ambiguous:

- prefer the limb indicated by metadata or extracted fields
- otherwise record ambiguity in `route_state`
- route to the most plausible limb and log the uncertainty
