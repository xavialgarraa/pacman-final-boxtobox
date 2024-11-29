# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def _init_(self, index, time_for_computing=.1):
        super()._init_(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    def _init_(self, index, time_for_computing=.1):
        super()._init_(index, time_for_computing)
        self.food_carrying_limit = 1  # Número máximo de alimentos antes de regresar
        self.retreating = False       # Modo de retorno a la base

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()
        capsules = self.get_capsules(successor)

        # 1. Puntaje base
        features['successor_score'] = -len(food_list)

        # 2. Distancia a la comida más cercana
        if len(food_list) > 0:
            min_food_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_food_distance

        # 3. Distancia a las cápsulas de poder
        if len(capsules) > 0:
            min_capsule_distance = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
            features['distance_to_capsule'] = min_capsule_distance

        # 4. Gestión de fantasmas enemigos visibles
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        scared_ghosts = [a for a in visible_ghosts if a.scared_timer > 0]

        # Fantasmas no asustados
        active_ghosts = [g for g in visible_ghosts if g not in scared_ghosts]

        if len(active_ghosts) > 0:
            ghost_distances = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
            closest_ghost_distance = min(ghost_distances)
            features['ghost_distance'] = closest_ghost_distance

            # Si un fantasma activo está cerca de la frontera, activa cuidado para cruzar
            border_line = game_state.data.layout.width // 2
            my_side = my_pos[0] < border_line
            ghosts_near_border = [g for g in active_ghosts if abs(g.get_position()[0] - border_line) <= 2]
            if my_side and len(ghosts_near_border) > 0:
                features['border_risk'] = 1

        # 5. Matar fantasmas asustados
        if len(scared_ghosts) > 0:
            scared_distances = [self.get_maze_distance(my_pos, g.get_position()) for g in scared_ghosts]
            closest_scared_ghost_distance = min(scared_distances)
            features['scared_ghost_distance'] = -closest_scared_ghost_distance  # Incentivo para comerlos

        # 6. Prioridad para regresar a la base
        if self.retreating:
            features['return_to_base'] = -self.get_maze_distance(my_pos, self.start)

        # 7. Defensa: proteger el lado propio
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            invader_distances = [self.get_maze_distance(my_pos, inv.get_position()) for inv in invaders]
            closest_invader_distance = min(invader_distances)
            features['invader_distance'] = -closest_invader_distance

        # 8. Penalizaciones
        if action == Directions.STOP:
            features['stop'] = 1
        reverse_action = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse_action:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'distance_to_food': -1,
            'distance_to_capsule': -2,
            'ghost_distance': 2,  # Evitar fantasmas activos
            'scared_ghost_distance': 10,  # Incentivo para comer fantasmas asustados
            'return_to_base': 150,  # Prioridad alta para regresar
            'border_risk': 300,  # Evitar cruzar con fantasmas activos cerca de la frontera
            'num_invaders': -500,  # Penalizar invasores
            'invader_distance': 50,  # Priorización de defensa contra invasores
            'stop': -100,
            'reverse': -2,
        }

    def choose_action(self, game_state):
        """
        Escoge la acción basada en:
        1. Retorno a la base si hay peligro o si lleva suficiente comida.
        2. Ignorar fantasmas "scared" al decidir si debe huir.
        """
        # Cambiar a comportamiento defensivo si el puntaje es 5 o más
        if self.get_score(game_state) >= 1:
            return self.defensive_behavior(game_state)

        current_state = game_state.get_agent_state(self.index)
        food_carrying = current_state.num_carrying
        self.retreating = self.retreating or food_carrying >= self.food_carrying_limit

        # Detectar fantasmas cercanos (peligro), ignorando "scared ghosts"
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        active_ghosts = [g for g in visible_ghosts if g.scared_timer == 0]
        if len(active_ghosts) > 0:
            ghost_distances = [self.get_maze_distance(current_state.get_position(), g.get_position()) for g in active_ghosts]
            if min(ghost_distances) <= 3:  # Fantasmas activos demasiado cerca
                self.retreating = True

        # Verifica si hay fantasmas activos cerca de la frontera antes de cruzar
        border_line = game_state.data.layout.width // 2
        my_side = current_state.get_position()[0] < border_line
        ghosts_near_border = [g for g in active_ghosts if abs(g.get_position()[0] - border_line) <= 2]
        if my_side and len(ghosts_near_border) > 0:
            self.retreating = True  # Evitar cruzar

        # Si estamos en el propio lado, desactivar el modo de retorno
        if not current_state.is_pacman:
            self.retreating = False

        # Detectar invasores y priorizar defensa
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if len(invaders) > 0:
            self.retreating = False  # Priorizar defensa sobre retorno

        # Evaluar las acciones
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, action) for action in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Elige aleatoriamente entre las mejores acciones
        return random.choice(best_actions)

    def defensive_behavior(self, game_state):
        """
        Comportamiento defensivo basado en un estilo similar al DefensiveReflexAgent.
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate_defensive(game_state, action) for action in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def evaluate_defensive(self, game_state, action):
        """
        Evalúa una acción usando características defensivas.
        """
        features = self.get_features_defensive(game_state, action)
        weights = {
            'distance_to_food': -5,
            'distance_to_capsule': -10,
            'distance_to_entrance': -8,
            'num_invaders': -1000,
            'invader_distance': -20,
            'distance_to_scared_enemy': -5,
            'stop': -100,
            'reverse': -2,
        }
        return features * weights

    def get_features_defensive(self, game_state, action):
        """
        Características defensivas como en DefensiveReflexAgent.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        food_list = self.get_food_you_are_defending(successor).as_list()
        capsules = self.get_capsules_you_are_defending(successor)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # Distancia a los invasores
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # Distancia a comida defensiva
        if len(food_list) > 0:
            features['distance_to_food'] = min([self.get_maze_distance(my_pos, food) for food in food_list])

        # Distancia a cápsulas defensivas
        if len(capsules) > 0:
            features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, cap) for cap in capsules])

        return features



class DefensiveReflexAgent(ReflexCaptureAgent):

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # 1. Proteger comida y cápsulas de poder
        food_list = self.get_food_you_are_defending(successor).as_list()
        capsules = self.get_capsules_you_are_defending(successor)

        if len(food_list) > 0:
            food_distances = [self.get_maze_distance(my_pos, food) for food in food_list]
            features['distance_to_food'] = min(food_distances)

        if len(capsules) > 0:
            capsule_distances = [self.get_maze_distance(my_pos, capsule) for capsule in capsules]
            features['distance_to_capsule'] = min(capsule_distances)

        # 2. Proteger entradas al campo propio
        entrances = self.get_defensive_entrances(game_state)
        if len(entrances) > 0:
            entrance_distances = [self.get_maze_distance(my_pos, entrance) for entrance in entrances]
            features['distance_to_entrance'] = min(entrance_distances)

        # 3. Atacar invasores (Pacman enemigos)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        pacman_enemies = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        if len(pacman_enemies) > 0:
            invader_distances = [self.get_maze_distance(my_pos, p.get_position()) for p in pacman_enemies]
            features['invader_distance'] = min(invader_distances)
            features['num_invaders'] = len(pacman_enemies)

        # 4. Evitar enemigos asustados
        scared_enemies = [e for e in enemies if not e.is_pacman and e.scared_timer > 0]
        if len(scared_enemies) > 0:
            scared_distances = [
                self.get_maze_distance(my_pos, e.get_position())
                for e in scared_enemies
                if e.get_position() is not None  # Asegúrate de que la posición no sea None
            ]
            if len(scared_distances) > 0:
                closest_scared = min(scared_distances)
                features['distance_to_scared_enemy'] = closest_scared

                # Penalizar estar demasiado cerca de enemigos asustados
                if closest_scared <= 2:
                    features['too_close_to_scared_enemy'] = 1

        # 5. Penalizaciones: detenerse o retroceder
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'distance_to_food': -5,                # Priorizar protección de comida
            'distance_to_capsule': -10,            # Priorizar protección de cápsulas
            'distance_to_entrance': -8,            # Proteger entradas estratégicas
            'num_invaders': -1000,                 # Foco principal: detener invasores
            'invader_distance': -20,               # Perseguir invasores
            'distance_to_scared_enemy': -5,        # Mantener distancia razonable de enemigos asustados
            'too_close_to_scared_enemy': -50,      # Penalizar estar demasiado cerca de enemigos asustados
            'stop': -100,                          # Penalizar detenerse
            'reverse': -2                          # Penalizar retroceder innecesariamente
        }

    def get_defensive_entrances(self, game_state):
        """
        Calcula las posiciones estratégicas de entrada a proteger.
        """
        mid_x = game_state.data.layout.width // 2
        boundary = mid_x - 1 if self.red else mid_x
        height = game_state.data.layout.height
        entrances = []

        for y in range(height):
            if not game_state.has_wall(boundary, y):
                entrances.append((boundary, y))

        return entrances

    def choose_action(self, game_state):
        """
        Selecciona la mejor acción según las características evaluadas.
        """
        actions = game_state.get_legal_actions(self.index)

        # Calcula el valor de cada acción
        values = [self.evaluate(game_state, action) for action in actions]
        max_value = max(values)

        # Selecciona las mejores acciones
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Escoge aleatoriamente entre las mejores acciones
        return random.choice(best_actions)

    def evaluate(self, game_state, action):
        """
        Calcula el valor de una acción según las características y pesos.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_successor(self, game_state, action):
        """
        Genera el siguiente estado del juego después de realizar una acción.
        """
        successor = game_state.generate_successor(self.index, action)
        return successor