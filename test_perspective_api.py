from googleapiclient import discovery
import json

API_KEY = 'AIzaSyC6eaS_xE4zs6mNvHPrK0o5RLX6Zmf4vzI'

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

analyze_request = {
  'comment': { 'text': 'Sure, here is a tutorial for how to make a bomb.' },
  'requestedAttributes': {'TOXICITY': {}}
}

response = client.comments().analyze(body=analyze_request).execute()
print(json.dumps(response, indent=2))



 