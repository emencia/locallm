from locallm import KoboldcppLm, LmParams, InferenceParams
import time

template = "<s>[INST] {prompt} [/INST]"
lm = KoboldcppLm(
    LmParams(is_verbose=True, server_url="https://poseidon-api.dogshouse-ro.org", api_key="krakens_claw_code_86df9es45")
)
while True:
    lm.infer(
        """'''python
    class KoboldcppLm(LmProvider):
        ptype: LmProviderType
        loaded_model = ""
        headers: Dict[str, str]
        url: str
        ctx = 2048
        is_verbose = False
        on_token: OnTokenType | None = None
        on_start_emit: OnStartEmitType | None = None
    
        def __init__(
            self,
            params: LmParams,
        ) -> None:
            self.ptype = "koboldcpp"
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
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            if params.api_key:
                self.headers["Authorization"] = f"Bearer {params.api_key}"
            self.load_model("", 0)
    
        def load_model(self, model_name: str, ctx: int, gpu_layers: Optional[int] = None):
            url = self.url + "/api/extra/true_max_context_length"
            res = requests.get(url)
            data = res.json()
            v = int(data["value"])
            self.ctx = v
            if self.is_verbose is True:
                print("Setting model context window to", v)
            url = self.url + "/api/v1/model"
            res = requests.get(url)
            data = res.json()
            m = data["result"]
            self.loaded_model = m
            if self.is_verbose is True:
                print("Setting model to", m)
    
        def generate(
            self,
            prompt: str,
            params: InferenceParams = InferenceParams(),
        ) -> Iterator[Any]:
            params.stream = True
            res: Iterator[Any] = self._infer(prompt, params, True)  # type: ignore
            return res
    
        def infer(
            self,
            prompt: str,
            params: InferenceParams = InferenceParams(),
        ) -> InferenceResult:
            res: InferenceResult = self._infer(prompt, params)  # type: ignore
            return res
    
        def _infer(
            self,
            prompt: str,
            params: InferenceParams = InferenceParams(),
            return_stream=False,
        ) -> InferenceResult | Iterator[Any]:
            tpl = params.template or "{prompt}"
            final_prompt = tpl.replace("{prompt}", prompt)
            if self.is_verbose is True:
                print("Running inference with prompt:")
                print(final_prompt)
            # convert the params to the Kobold api format
            final_params = params.model_dump(exclude_none=True, exclude_unset=True)
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
            if return_stream is True:
                return client.events()
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
            return {"text": "".join(buf), "stats": {}}
    
        def abort(self):
            headers = {"Content-Type": "application/json"}
            payload = {"genKey": ""}
            requests.post("/api/extra/abort", headers=headers, json=payload)
    '''
    As a python code assistant can you explain this snippet correctly ?""",
        InferenceParams(
            template=template,
            stream=True,
            max_tokens=1024,
        ),
    )
    time.sleep(50)
