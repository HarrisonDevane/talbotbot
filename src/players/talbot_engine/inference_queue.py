import torch
import torch.nn.functional as F
from collections import deque
import threading
import time

class InferenceQueue:

    def __init__(self, model_player, logger, batch_size=32):
        """
        Initialize the inference queue and start the background worker.

        Parameters
        ----------
        model_player : object
            A model wrapper with a get_policy_value() method.
        logger : logging.Logger
            Logger for outputting status or debug information.
        batch_size : int, optional
            Maximum number of items to batch at once (default is 32).
        """
        self.model_player = model_player
        self.logger = logger
        self.batch_size = batch_size

        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        self.queue = deque()
        self.results = {}

        self.running = True
        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """
        Stop the inference worker thread and clean up resources.
        """
        with self.condition:
            self.running = False
            self.condition.notify_all()
        self.worker_thread.join()

    def queue_inference(self, node_id, board_tensor):
        """
        Add a new inference task to the queue.

        Parameters
        ----------
        node_id : Any hashable
            Unique identifier for the node (used to fetch result later).
        board_tensor : torch.Tensor
            Tensor representation of the board state to be evaluated.
        """
        with self.condition:
            self.queue.append((node_id, board_tensor))
            self.condition.notify()

    def get_result(self, node_id, timeout=1.0):
        """
        Retrieve the inference result for a given node.

        Parameters
        ----------
        node_id : Any hashable
            The ID corresponding to a previously queued inference.
        timeout : float, optional
            Maximum time to wait in seconds (default is 1.0).

        Returns
        -------
        tuple or None
            A tuple of (policy_probs, value) if available within timeout,
            otherwise None.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if node_id in self.results:
                    return self.results.pop(node_id)
            time.sleep(0.01)
        return None

    def _inference_worker(self):
        """
        Internal method: Background thread for batching and executing inference.

        Continuously monitors the queue and runs the model in batches.
        Results are stored in the `results` dictionary using node_id as keys.
        """
        while True:
            with self.condition:
                while self.running and len(self.queue) < self.batch_size:
                    self.condition.wait(timeout=0.01)
                if not self.running and not self.queue:
                    break
                batch = [self.queue.popleft() for _ in range(min(self.batch_size, len(self.queue)))]

            if not batch:
                continue

            node_ids, board_tensors = zip(*batch)
            input_tensor = torch.stack(board_tensors)
            policy_logits, value_output = self.model_player.get_policy_value(input_tensor)
            policy_probs = F.softmax(policy_logits, dim=1)

            with self.lock:
                for i, node_id in enumerate(node_ids):
                    self.results[node_id] = (policy_probs[i], value_output[i].item())
