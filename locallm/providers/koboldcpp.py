from typing import Dict
import json
import sseclient
import requests
from ..schemas import InferenceParams, LmParams, OnTokenType, OnStartEmitType
from ..provider import LmProvider, defaultOnToken


class Koboldcpp(LmProvider):
    loaded_model = ""
    headers: Dict[str, str]
    url: str
    ctx = 8192
    is_verbose = False
    on_token: OnTokenType
    on_start_emit: OnStartEmitType

    def __init__(
        self,
        params: LmParams,
    ) -> None:
        if params.server_url is None:
            self.url = "http://localhost:5001"
            print(
                "No server url provided: using the default local one: "
                "http://localhost:5001"
            )
        else:
            self.url = params.server_url
        if params.is_verbose is True:
            self.is_verbose = True
        if params.on_token:
            self.on_token = params.on_token
        else:
            self.on_token = defaultOnToken
        if params.on_start_emit:
            self.on_start_emit = params.on_start_emit
        self.headers = {
            # "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def load_model(self, model_name: str, ctx: int):
        if self.is_verbose is True:
            print("Setting model context window to", ctx)
        self.ctx = ctx

    def infer(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
    ) -> str:
        tpl = params.template or "{prompt}"
        final_prompt = tpl.replace("{prompt}", prompt)
        if self.is_verbose is True:
            print("Running inference with prompt:")
            print(final_prompt)
        # convert the params to the Kobold api format
        final_params = params.model_dump(exclude_none=True, exclude_unset=True)
        if "tfs_z" in final_params:
            final_params["tfs"] = final_params["tfs_z"]
            del final_params["tfs_z"]
        if "stop" in final_params:
            final_params["stop_sequence"] = final_params["stop"]
            del final_params["stop"]
        if "repeat_penalty" in final_params:
            final_params["rep_pen"] = final_params["repeat_penalty"]
            del final_params["repeat_penalty"]
        if "presence_penalty" in final_params:
            del final_params["presence_penalty"]
        if "frequency_penalty" in final_params:
            del final_params["frequency_penalty"]
        if "threads" in final_params:
            del final_params["threads"]
        if "max_tokens" in final_params:
            final_params["max_length"] = final_params["max_tokens"]
            del final_params["max_tokens"]
        if "stream" in final_params:
            del final_params["stream"]
        if self.is_verbose is True:
            print("Inference parameters:")
            print(final_params)
        # run the query
        payload = {
            "prompt": final_prompt,
            "max_context_length": self.ctx,
            **final_params,
        }
        url = self.url + "/api/extra/generate/stream"
        response = requests.post(url, stream=True, headers=self.headers, json=payload)
        client = sseclient.SSEClient(response)  # type: ignore
        buf = []
        i = 0
        for event in client.events():
            # print(event)
            if i == 0:
                if self.on_start_emit:
                    self.on_start_emit(None)
            # print("|begin|", event, "|end|")
            data = json.loads(event.data)
            # print(data)
            if self.on_token:
                self.on_token(data["token"])
            buf.append(data["token"])
            i += 1
        return "".join(buf)
