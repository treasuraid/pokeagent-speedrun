"""
Agent modules for Pokemon Emerald speedrunning agent
"""

from utils.vlm import VLM
from .action import action_step
from .memory import memory_step
from .perception import perception_step
from .planning import planning_step
from .simple import SimpleAgent, get_simple_agent, simple_mode_processing_multiprocess, configure_simple_agent_defaults


class Agent:
    """
    Unified agent interface that encapsulates all agent logic.
    The client just calls agent.step(game_state) and gets back an action.
    """
    
    def __init__(self, args=None):
        """
        Initialize the agent based on configuration.
        
        Args:
            args: Command line arguments with agent configuration
        """
        # Extract configuration
        backend = args.backend if args else "gemini"
        model_name = args.model_name if args else "gemini-2.5-flash"
        simple_mode = args.simple if args else False
        
        # Initialize VLM
        self.vlm = VLM(backend=backend, model_name=model_name)
        print(f"   VLM: {backend}/{model_name}")
        
        # Initialize agent mode
        self.simple_mode = simple_mode
        if simple_mode:
            # Use global SimpleAgent instance to enable checkpoint persistence
            self.simple_agent = get_simple_agent(self.vlm)
            print(f"   Mode: Simple (direct frame->action)")
        else:
            # Four-module agent context
            self.context = {
                'perception_output': None,
                'planning_output': None,
                'memory': []
            }
            print(f"   Mode: Four-module architecture")
    
    def step(self, game_state):
        """
        Process a game state and return an action.
        
        Args:
            game_state: Dictionary containing:
                - screenshot: PIL Image
                - game_state: Dict with game memory data
                - visual: Dict with visual observations
                - audio: Dict with audio observations
                - progress: Dict with milestone progress
        
        Returns:
            dict: Contains 'action' and optionally 'reasoning'
        """
        print("\n➡️ Agent step processing...")
        print("Game state keys:", list(game_state.keys()))

        # pretty print game_state
        print("Game state summary:")
        for key, value in game_state.items():
            if key != 'screenshot':
                print(f" - {key}: {type(value)}")
            else:
                print(f" - {key}: PIL Image")
        
        if self.simple_mode:
            # Simple mode - delegate to SimpleAgent
            return self.simple_agent.step(game_state)
        else:
            # Four-module processing
            try:
                # Extract key components from game_state
                screenshot = game_state.get('frame')  # The key is 'frame', not 'screenshot'
                state_data = game_state

                # Initialize tracking variables if not present
                if 'observation_buffer' not in self.context:
                    self.context['observation_buffer'] = []
                if 'recent_actions' not in self.context:
                    self.context['recent_actions'] = []
                if 'current_plan' not in self.context:
                    self.context['current_plan'] = None
                if 'memory_context' not in self.context:
                    self.context['memory_context'] = ""
                if 'frame_id' not in self.context:
                    self.context['frame_id'] = 0

                self.context['frame_id'] += 1

                # 1. Perception - understand what's happening
                # perception_step(frame, state_data, vlm) -> (observation, slow_thinking_needed)
                observation, slow_thinking_needed = perception_step(
                    screenshot,
                    state_data,
                    self.vlm
                )
                self.context['perception_output'] = observation

                # Add observation to buffer
                self.context['observation_buffer'].append({
                    'frame_id': self.context['frame_id'],
                    'observation': observation,
                    'state': state_data
                })

                # 2. Planning - decide strategy
                # planning_step(memory_context, current_plan, slow_thinking_needed, state_data, vlm)
                planning_output = planning_step(
                    self.context['memory_context'],
                    self.context['current_plan'],
                    slow_thinking_needed,
                    state_data,
                    self.vlm
                )
                self.context['planning_output'] = planning_output
                self.context['current_plan'] = planning_output

                # 3. Memory - update context
                # memory_step(memory_context, current_plan, recent_actions, observation_buffer, vlm)
                memory_output = memory_step(
                    self.context['memory_context'],
                    self.context['current_plan'],
                    self.context['recent_actions'],
                    self.context['observation_buffer'],
                    self.vlm
                )
                self.context['memory_context'] = memory_output

                # 4. Action - choose button press
                # action_step(memory_context, current_plan, latest_observation, frame, state_data, recent_actions, vlm)
                action_output = action_step(
                    self.context['memory_context'],
                    self.context['current_plan'],
                    observation,
                    screenshot,
                    state_data,
                    self.context['recent_actions'],
                    self.vlm
                )

                # Store actions in recent_actions for next iteration
                if action_output:
                    if isinstance(action_output, list):
                        self.context['recent_actions'].extend(action_output)
                    else:
                        self.context['recent_actions'].append(action_output)
                    # Keep only last 20 actions
                    self.context['recent_actions'] = self.context['recent_actions'][-20:]

                # Clear observation buffer periodically (keep last 5)
                if len(self.context['observation_buffer']) > 5:
                    self.context['observation_buffer'] = self.context['observation_buffer'][-5:]

                # Return in the format expected by client: {'action': [...]}
                return {'action': action_output} if action_output else None
                
            except Exception as e:
                print(f"❌ Agent error: {e}")
                return None


__all__ = [
    'Agent',
    'action_step',
    'memory_step', 
    'perception_step',
    'planning_step',
    'SimpleAgent',
    'get_simple_agent',
    'simple_mode_processing_multiprocess',
    'configure_simple_agent_defaults'
]