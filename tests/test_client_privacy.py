"""No-network tests that provider errors and argv do not disclose request secrets."""
import io
import sys
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'extraction_pipeline/code'))
from openai_compatible_client import OpenAICompatibleClient, ApiCallError


class ClientPrivacyTests(unittest.TestCase):
    def test_upload_secret_not_in_argv(self):
        client=OpenAICompatibleClient('https://example.invalid','SYNTHETIC_SECRET','synthetic')
        with patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='{"id":"synthetic-file"}')) as run:
            self.assertEqual(client.upload_batch_file(['{}']),'synthetic-file')
        args,kwargs=run.call_args
        self.assertNotIn('SYNTHETIC_SECRET',' '.join(args[0]))
        self.assertEqual(kwargs['input'],'Authorization: Bearer SYNTHETIC_SECRET\n')

    def test_response_body_redacted(self):
        client=OpenAICompatibleClient('https://example.invalid','SYNTHETIC_SECRET','synthetic')
        err=urllib.error.HTTPError('https://example.invalid',400,'bad',{},io.BytesIO(b'PRIVATE_REQUEST_ECHO'))
        with patch('urllib.request.urlopen',side_effect=err), self.assertRaises(ApiCallError) as raised:
            client._post_json('/synthetic',{})
        self.assertNotIn('PRIVATE_REQUEST_ECHO',str(raised.exception))


if __name__=='__main__':unittest.main()
