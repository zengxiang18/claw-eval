import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import requests
from omegaconf import OmegaConf
from pathlib import Path
import json
import copy
import logging
import sys
from typing import Any, Optional
from contextlib import contextmanager

EXECUTION_ENV_ROOT = Path("/mnt/z00945507")

CONFIG_PATH = EXECUTION_ENV_ROOT / "7.242.109.44.5443.config.yaml"
DOWNLOAD_PATH = EXECUTION_ENV_ROOT / "downloads"
config = OmegaConf.load(CONFIG_PATH)
endpoint= "7.242.109.44:7443"

project_id: str = "openclaw_win11"
user_id: str = "win11-demo-test"


class Win11BatchLogger:
    """统一的日志管理器"""

    def __init__(self, script_name: str = "win11_batch_run.py"):
        self.script_name = script_name
        self._start_times = {}  # 记录操作开始时间
        self._setup_logging()

    def _setup_logging(self):
        """配置logging基础设置"""
        # 确保stdout使用utf-8编码
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')

        self.formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.logger = logging.getLogger('Win11Batch')
        self.logger.setLevel(logging.DEBUG)

        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def _format_result(self, result: Any) -> str:
        """格式化结果为一行显示"""
        if result is None:
            return ""
        if isinstance(result, dict):
            items = [f"{k}={v}" for k, v in result.items()]
            return "{" + ", ".join(items) + "}"
        if isinstance(result, (list, tuple)):
            return "[" + ", ".join(str(x) for x in result) + "]"
        return str(result)

    def _log(self, level: int, operation: str, message: str,
             command: str = None, result: Any = None, duration: float = None):
        """核心日志方法"""
        parts = [operation]
    
        if command:
            parts.append(f"cmd=[{command}]")

        parts.append(message)

        if duration is not None:
            parts.append(f"duration={duration:.2f}s")

        if result is not None:
            parts.append(f"result={self._format_result(result)}")

        log_msg = " | ".join(parts)
        self.logger.log(level, log_msg)

    def op_start(self, operation: str, detail: str = "", command: str = None) -> str:
        """记录操作开始，返回操作ID用于计算耗时"""
        op_id = f"{operation}_{time.time()}"
        self._start_times[op_id] = time.time()

        msg = f"START: {detail}" if detail else "START"
        self._log(logging.INFO, operation, msg, command=command)
        return op_id

    def op_end(self, op_id: str, operation: str, success: bool,
               detail: str = "", result: Any = None):
        """记录操作结束"""
        duration = None
        if op_id in self._start_times:
            duration = time.time() - self._start_times.pop(op_id)

        status = "SUCCESS" if success else "FAILED"
        msg = f"{status}: {detail}"
        self._log(logging.INFO if success else logging.ERROR,
                  operation, msg, result=result, duration=duration)

    def info(self, operation: str, message: str):
        self._log(logging.INFO, operation, message)

    def error(self, operation: str, message: str, error: Any = None):
        self._log(logging.ERROR, operation, message, result=error)

    def debug(self, operation: str, message: str):
        self._log(logging.DEBUG, operation, message)

    def http_request(self, method: str, url: str, body: dict = None):
        """记录HTTP请求"""
        if body:
            body_str = json.dumps(body, ensure_ascii=False)
            self._log(logging.INFO, "HTTP", f"{method} {url} body={body_str}")
        else:
            self._log(logging.INFO, "HTTP", f"{method} {url}")

    def http_response(self, status_code: int, response_text: str = None, truncated: int = 500):
        """记录HTTP响应"""
        try:
            response_json = json.loads(response_text) if response_text else None
        except (json.JSONDecodeError, TypeError):
            response_json = None

        level = logging.INFO if status_code == 200 else logging.ERROR

        if response_json:
            self._log(level, "HTTP", f"Response [{status_code}]", result=response_json)
        else:
            text = response_text[:truncated] if response_text else ""
            self._log(level, "HTTP", f"Response [{status_code}] result={text}")

    def stream_output(self, line: str, is_raw: bool = False):
        """记录流式输出"""
        prefix = "[RAW]" if is_raw else "[STREAM]"
        # 直接打印，避免unicode转义
        sys.stdout.write(f"  {prefix} {line}\n")
        sys.stdout.flush()

    @contextmanager
    def track_command(self, operation: str, command: str):
        """上下文管理器：自动追踪命令执行时间"""
        op_id = self.op_start(operation, detail="", command=command)
        try:
            yield op_id
            self.op_end(op_id, operation, True)
        except Exception as e:
            self.op_end(op_id, operation, False, detail=str(e))
            raise

    def task_start(self, json_file: str):
        """任务开始"""
        self._log(logging.INFO, "TASK",
                  f"START | script={self.script_name} | json_file={json_file}")

    def task_complete(self, duration: float):
        """任务完成"""
        self._log(logging.INFO, "TASK", f"COMPLETE", duration=duration)

    def task_failed(self, reason: str):
        """任务失败"""
        self._log(logging.ERROR, "TASK", f"FAILED | reason={reason}")


# 全局logger实例
logger = Win11BatchLogger("win11_batch_run.py")


def close_pod(env_id: str, index: int = 0):
    op_id = logger.op_start("CLOSE_POD", f"env_id={env_id}")

    url = f"http://{endpoint}/{project_id}/{user_id}/v1/env/gem/close"
    logger.http_request("POST", url, {"env_id": env_id})

    response = requests.post(url=url, json={"env_id": env_id}, timeout=120)
    logger.http_response(response.status_code, response.text)

    if response.status_code != 200:
        try:
            error_detail = response.json()
        except:
            error_detail = response.text
        logger.op_end(op_id, "CLOSE_POD", False,
                      detail="", result=error_detail)
    else:
        logger.op_end(op_id, "CLOSE_POD", True, detail=f"env_id={env_id}")


def create_pod(index: int = 0):
    op_id = logger.op_start("CREATE_POD", "Creating Windows11 sandbox")

    url = f"http://{endpoint}/{project_id}/{user_id}/v1/env/gem/make"
    runtime = copy.deepcopy(config.sandbox_runtimes.windows.runtime)
    runtime.spec.containers[0].image = "csb-private-swr-on5ehj.swr-pro.myhuaweicloud.com/ai-siye-pipeline/win11:0.0.1"
    body = {
        'wait_for_ready': True,
        'wait_timeout': 1000,
        'args': {
            'resources': {
                'cpu_request': '6',
                'cpu_limit': '8',
                'memory_request': '20Gi',
                'memory_limit': '32Gi',
                'npu_request': '0.0',
                'npu_limit': '0.0',
                'gpu_request': '0.0',
                'gpu_limit': '0.0'
            },
            'runtime_config': OmegaConf.to_container(runtime, resolve=True)
        },
        'env_id': str(uuid.uuid4()),
        'runtime_type': 'windows'
    }

    logger.http_request("POST", url, body)

    pod_response = requests.post(url=url, json=body, timeout=180)
    logger.http_response(pod_response.status_code, pod_response.text)

    if pod_response.status_code != 200:
        try:
            error_detail = pod_response.json()
        except:
            error_detail = pod_response.text
        logger.op_end(op_id, "CREATE_POD", False, detail="", result=error_detail)
        raise RuntimeError(f"Create pod failed: {pod_response.status_code}")

    env_id = pod_response.json()["env_id"]
    logger.op_end(op_id, "CREATE_POD", True, detail=f"env_id={env_id}")
    return env_id


def upload_json_file(env_id: str, json_file_path: str, remote_path: str, index: int = 0):
    op_id = logger.op_start("UPLOAD", f"{json_file_path} -> {remote_path}")
    url = f"http://{endpoint}/{project_id}/{user_id}/v1/env/gem/upload_file"

    try:
        with open(json_file_path, 'rb') as f:
            files = {'file': (os.path.basename(json_file_path), f)}
            params = {'env_id': env_id, 'remote_path': remote_path}
            response = requests.post(url=url, files=files, data=params, timeout=120)

        logger.http_response(response.status_code, response.text)

        if response.status_code != 200:
            logger.op_end(op_id, "UPLOAD", False, detail=response.text)
            return

        logger.op_end(op_id, "UPLOAD", True, detail=f"{json_file_path} -> {remote_path}")

    except Exception as e:
        logger.op_end(op_id, "UPLOAD", False, detail=str(e))


def download_folder_zip(env_id: str, index: int = 0):
    remote_path = fr"C:/Users/p_panguRL/.openclaw/agents.zip"
    file_path = DOWNLOAD_PATH / f"{args.json_file.removesuffix('.json')}.zip"

    op_id = logger.op_start("DOWNLOAD", f"{remote_path} -> {file_path}")
    url = f"http://{endpoint}/{project_id}/{user_id}/v1/env/gem/download_file"

    try:
        params = {'env_id': env_id, 'remote_path': remote_path}
        response = requests.post(url=url, data=params, timeout=120)

        if response.status_code != 200:
            logger.http_response(response.status_code, response.text)
            logger.op_end(op_id, "DOWNLOAD", False, detail=response.text)
            return

        with open(file_path, 'wb') as f:
            f.write(response.content)
        logger.op_end(op_id, "DOWNLOAD", True, detail=f"{remote_path} -> {file_path}")

    except Exception as e:
        logger.op_end(op_id, "DOWNLOAD", False, detail=str(e))


def exec_command(env_id: str, cmd: str = "dir"):
    op_id = logger.op_start("EXEC", f"command={cmd}")

    url = f"http://{endpoint}/{project_id}/{user_id}/v1/env/gem/extend"
    body = {
        "cmd_name": "exec_command",
        "env_id": env_id,
        "args": {
            "command": [cmd],
        }
    }
    logger.http_request("POST", url, body)

    pod_response = requests.post(url=url, json=body, timeout=90)
    logger.http_response(pod_response.status_code, pod_response.text)

    if pod_response.status_code != 200:
        try:
            error_detail = pod_response.json()
        except:
            error_detail = pod_response.text
        logger.op_end(op_id, "EXEC", False, detail="", result=error_detail)
        raise RuntimeError(f"exec_command failed: {pod_response.status_code}")

    result = pod_response.json()
    logger.op_end(op_id, "EXEC", True, result=result.get("response"))
    return result


def exec_stream_command(env_id: str, command: str, timeout_seconds: int = 1800):
    op_id = logger.op_start("EXEC_STREAM", command=command)
    start_time = time.time()

    url = f"http://{endpoint}/{project_id}/{user_id}/v1/env/gem/stream"
    body = {
        "cmd_name": "exec_command",
        "env_id": env_id,
        "args": {
            "command": [command],
            "timeout": timeout_seconds
        }
    }
    logger.http_request("POST", url, body)
    timed_out = False

    try:
        with requests.post(url=url, json=body, timeout=7200, stream=True) as response:
            logger.http_response(response.status_code)

            if response.status_code != 200:
                logger.op_end(op_id, "run_code failed", False, detail=response.text)
                close_pod(env_id)
                return None

            for line in response.iter_lines():
                # 检查是否超时
                if (time.time() - start_time) > timeout_seconds:
                    timed_out = True
                    logger.error("EXEC_TIMEOUT", f"Command exceeded {timeout_seconds}s, terminating...")
                    break

                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        data = json.loads(decoded_line)
                        output = data.get('output', data)
                        if isinstance(output, str):
                            output = output.replace('\\n', '\n')
                        logger.stream_output(str(output))
                    except json.JSONDecodeError:
                        logger.stream_output(decoded_line, is_raw=True)
        
        if timed_out:
            # 强制终止远程进程
            kill_cmd = f'taskkill /F /IM python.exe'
            exec_command(env_id, kill_cmd)
            logger.op_end(op_id, "EXEC_STREAM", False, detail=f"Timed out after {timeout_seconds}s")
            return None

        logger.op_end(op_id, "EXEC_STREAM", True)

    except requests.exceptions.RequestException as e:
        logger.op_end(op_id, "EXEC_STREAM", False, detail=str(e))
        close_pod(env_id)


def exec_stream_command1(env_id: str, command: str):
    op_id = logger.op_start("EXEC_STREAM", command=command)

    url = f"http://{endpoint}/{project_id}/{user_id}/v1/env/gem/stream"
    body = {
        "cmd_name": "exec_command",
        "env_id": env_id,
        "args": {
            "command": [command],
            "timeout": 180
        }
    }
    logger.http_request("POST", url, body)

    final_result = None

    try:
        with requests.post(url=url, json=body, timeout=7200, stream=True) as response:
            logger.http_response(response.status_code)

            if response.status_code != 200:
                logger.op_end(op_id, "EXEC_STREAM", False, detail=response.text)
                close_pod(env_id)
                return None

            for line in response.iter_lines():
                if not line:
                    continue

                decoded_line = line.decode("utf-8", errors="replace").strip()

                if decoded_line.startswith("event:"):
                    logger.stream_output(decoded_line, is_raw=True)
                    continue

                if decoded_line.startswith("data:"):
                    decoded_line = decoded_line[len("data:"):].strip()

                try:
                    data = json.loads(decoded_line)
                    output = data.get("output", data)

                    if isinstance(output, str):
                        output = output.replace("\\n", "\n")

                    logger.stream_output(str(output))

                    if not isinstance(data, dict) or not data:
                        continue

                    if "status_code" in data and "response" in data:
                        resp = data.get("response", {})
                        final_result = {
                            "status_code": data.get("status_code"),
                            "env_id": data.get("env_id"),
                            "command": resp.get("command"),
                            "exit_code": resp.get("exit_code"),
                            "stdout": resp.get("stdout", ""),
                            "stderr": resp.get("stderr", ""),
                            "pid": resp.get("pid"),
                            "elapsed_seconds": resp.get("elapsed_seconds"),
                            "success": resp.get("success"),
                            "heartbeat": resp.get("heartbeat"),
                        }

                except json.JSONDecodeError:
                    logger.stream_output(decoded_line, is_raw=True)

        logger.op_end(op_id, "EXEC_STREAM", True, result=final_result)

    except requests.exceptions.RequestException as e:
        logger.op_end(op_id, "EXEC_STREAM", False, detail=str(e))
        close_pod(env_id)
        return None

    return final_result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    args = parser.parse_args()

    def main(idx):
        env_id = ""
        start_time = time.time()
        try:
            logger.task_start(args.json_file)

            env_id = create_pod(index=idx)

            ipp = r'netstat -ano | findstr :18789'

            # cmd1 = r'cd C:\Users\p_panguRL\.openclaw && ren openclaw.json oppp.json'
            # exec_stream_command(env_id, cmd1)

            upload_json_file(env_id, "openclaw.json", r"C:\Users\p_panguRL\.openclaw\openclaw.json")
            upload_json_file(env_id, "time.py", r"C:\Users\p_panguRL\OpenClaw-Pipeline-master\openclaw-task\time.py")
            upload_json_file(env_id, "openclaw_automation.py", r"C:\Users\p_panguRL\zengxiang\openclaw-task\openclaw_automation.py")

            kill_open_claw_cmd = r'taskkill /IM node.exe /F'
            launch_open_claw_cmd = r'chcp 65001 && set PYTHONIOENCODING=utf-8 &&  C:\Users\p_panguRL\AppData\Roaming\npm\openclaw.cmd gateway start'

            exec_stream_command(env_id, kill_open_claw_cmd)
            exec_stream_command(env_id, launch_open_claw_cmd)

            change_cmd = fr'chcp 65001 && set PYTHONIOENCODING=utf-8 &&  cd C:\Users\p_panguRL\OpenClaw-Pipeline-master\openclaw-task && C:\Users\p_panguRL\AppData\Local\Programs\Python\Python314\python.exe  time.py'
            exec_stream_command(env_id, change_cmd)

            cmd2 = r'setx no_proxy "localhost,127.0.0.1,7.150.9.169,10.90.91.214" /M  && setx NO_PROXY "localhost,127.0.0.1,7.150.9.169,10.90.91.214" /M'
            exec_stream_command(env_id, cmd2)

            flag = False

            for i in range(3):
                result = exec_stream_command1(env_id, ipp)
                if result and result.get("status_code") == 200 and result.get("success") is True and result.get("exit_code") == 0:
                    flag = True
                    logger.info("探测成功", "Port 18789 detected successfully")
                    break
                else:
                    logger.info("尝试重启openclaw", f"Attempt {i+1}/3 failed, retrying...")
                    cc_open_claw_cmd = r'chcp 65001 && set PYTHONIOENCODING=utf-8 &&  C:\Users\p_panguRL\AppData\Roaming\npm\openclaw.cmd gateway status'

                    exec_stream_command(env_id, kill_open_claw_cmd)
                    exec_stream_command(env_id, launch_open_claw_cmd)
                    time.sleep(8)

            if flag:
                open_claw_task_cmd = fr'chcp 65001 && set PYTHONIOENCODING=utf-8 &&  cd C:\Users\p_panguRL\zengxiang\openclaw-task &&  C:\Users\p_panguRL\AppData\Local\Programs\Python\Python314\python.exe openclaw_automation.py --config {args.json_file}'
                # open_claw_task_cmd1 = fr'chcp 65001 && set PYTHONIOENCODING=utf-8 &&  cd C:\Users\p_panguRL\zengxiang\openclaw-task &&  C:\Users\p_panguRL\AppData\Local\Programs\Python\Python314\python.exe openclaw_automation.py configs\config_simple.json'
                open_claw_cmd2 = fr'cd C:\Users\p_panguRL\.openclaw && tar -a -c -f agents.zip agents'

                exec_stream_command(env_id, open_claw_task_cmd, 3600) # 设置稍微大一点，根据命令设置超时时间
                exec_stream_command(env_id, open_claw_cmd2, 180)
                exec_stream_command(env_id, ipp, 180)

                logger.info("DOWNLOAD", "Starting download agents.zip")
                download_folder_zip(env_id, index=idx)
            else:
                logger.task_failed("沙箱创建失败 - port 18789 not responding after 3 attempts")

        except Exception as e:
            logger.task_failed(str(e))
            raise
        finally:
            if env_id:
                close_pod(env_id, index=idx)

        logger.task_complete(time.time() - start_time)


    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(main, i) for i in range(1)]
        for future in futures:
            future.result()