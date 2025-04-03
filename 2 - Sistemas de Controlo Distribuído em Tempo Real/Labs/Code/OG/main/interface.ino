#define MAYBE_PRINT_CLIENT_ID(cid) {if (cid) {Serial.print(cid); Serial.print(" ");}}
#define MAYBE_ADD_CLIENT_ID(cid, sz) {if (cid) {memcpy(data + sz, &cid, sizeof(cid));}}
#define MAYBE_ADD_CLIENT_SIZE(cid, sz) cid ? sz + sizeof(cid) : sz

#include "set_remove.h"

void interface(char *buffer) {
  char cmd, subcmd, x;
  int i;
  double val;
  uint8_t data[8] = {0};
  unsigned short client_id = (unsigned short) atoll(buffer);
  {
    size_t j = 0, size = strlen(buffer);
    while (j < size && (buffer[j] >= '0' && buffer[j] <= '9' || buffer[j] == ' '))  j++;
    if (j)
      memmove(buffer, buffer + j, size - j);
  }

  cmd = buffer[0];  
  switch (cmd) {
    case 'd': /* Set directly the duty cycle of the LED at luminaire i */
      std::sscanf(buffer, "%c %d %lf", &cmd, &i, &val);
      if (LUMINAIRE == i) {
        serial_duty_cycle = val;
        analogWrite(LED_PIN, serial_duty_cycle * DAC_RANGE);
        MAYBE_PRINT_CLIENT_ID(client_id);
        Serial.println("ack");
      }
      else {
        float aux = (float) val;
        memcpy(data, &aux, sizeof(aux));
        MAYBE_ADD_CLIENT_ID(client_id, sizeof(aux));
        enqueue_message(i, msg_t::SET_DUTY_CYCLE, data, MAYBE_ADD_CLIENT_SIZE(client_id, sizeof(aux)));
      }
      break;

    case 'g':
      std::sscanf(buffer, "%c %c %d", &cmd, &subcmd, &i);
      switch (subcmd) {
        case 'd': /* Get current duty cycle at luminaire i */
          if (LUMINAIRE == i){
            MAYBE_PRINT_CLIENT_ID(client_id);
            if (controller.get_feedback())
              Serial.printf("d %d %lf\n", LUMINAIRE, controller.get_duty_cycle());
            else
              Serial.printf("d %d %lf\n", LUMINAIRE, serial_duty_cycle);
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_DUTY_CYCLE, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;

        case 'e': /* Get average energy consumption at desk <i> since the last system restart. */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("e %d %lf\n", LUMINAIRE, energy);
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_ENERGY, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;
        
        case 'v': /* Get average visibility error at desk <i> since last system restart. */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("v %d %lf\n", LUMINAIRE, visibility_error / (double) iteration_counter);
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_VISIBILITY_ERROR, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;

        case 'f': /* Get the average flicker error on desk <i> since the last system restart. */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("f %d %lf\n", LUMINAIRE, flicker_error / (double) iteration_counter);
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_FLICKER_ERROR, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;

        case 'r': /* Get current illuminance reference at luminaire i */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("r %d %lf\n", LUMINAIRE, r);
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_REFERENCE, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;
        
        case 'l': /* Get measured illuminance at luminaire i */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("l %d %lf\n", LUMINAIRE, lux_value);
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_LUMINANCE, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;

        case 'o': /* Get current occupancy state at desk <i> */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("o %d %d\n", LUMINAIRE, node.get_occupancy());
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_OCCUPANCY, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;

        case 'a': /* Get anti-windup state at desk <i> */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("a %d %d\n", LUMINAIRE, controller.get_anti_windup());
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_ANTI_WINDUP, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;

        case 'k': /* Get feedback state at desk <i> */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("k %d %d\n", LUMINAIRE, controller.get_feedback());
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_FEEDBACK, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;

        case 'x': /* Get current external illuminance at desk <i> */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("x %d %lf\n", LUMINAIRE, lux_value - coupling_gains[LUMINAIRE] * duty_cycle);
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_EXTERNAL_LUMINANCE, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;

        case 'p': /* Get instantaneous power consumption at desk <i> */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("p %d %lf\n", LUMINAIRE, (controller.get_u() / DAC_RANGE) * MAXIMUM_POWER);
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);            
            enqueue_message(i, msg_t::GET_POWER, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;  

        case 't': /* Get elapsed time since last restart */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("t %d %lf\n", LUMINAIRE, micros() * pow(10, -6));
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_ELAPSED_TIME, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));
          }
          break;  
        
        case 'b': /* Get last minute buffer of variable <x> of desk <i>. <x> can be “l” or “d”. */
          std::sscanf(buffer, "%c %c %c %d", &cmd, &subcmd, &x, &i);
          if (LUMINAIRE == i) {
            if (x == 'l')
              buffer_l[LUMINAIRE] = !buffer_l[LUMINAIRE];
            else if (x == 'd')
              buffer_d[LUMINAIRE] = !buffer_d[LUMINAIRE];
            buffer_read_size = last_minute_buffer.get_used_space();
            buffer_read_counter = 0;
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("b %c %d ", x, LUMINAIRE);
          }
          else {
            memcpy(data, &x, sizeof(x));
            MAYBE_ADD_CLIENT_ID(client_id, sizeof(x));
            enqueue_message(i, msg_t::GET_BUFFER, data, MAYBE_ADD_CLIENT_SIZE(client_id, sizeof(x)));
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("b %c %d ", x, i);
          }
          break; 

        case 'O': /* Get lower bound on illuminance for Occupied state at desk <i> */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("O %d %lf\n", LUMINAIRE, node.get_lower_bound_occupied());
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_LOWER_BOUND_OCCUPIED, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));     
          }     
          break; 

        case 'U': /* Get lower bound on illuminance for Unoccupied state at desk <i> */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("U %d %lf\n", LUMINAIRE, node.get_lower_bound_unoccupied());
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_LOWER_BOUND_UNOCCUPIED, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));   
          }       
          break;

        case 'L': /* Get current illuminance lower bound at desk <i> */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("L %d %lf\n", LUMINAIRE, node.get_lower_bound());
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_LOWER_BOUND, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0));   
          }       
          break;

        case 'c': /* Set current energy cost at desk <i> */
          if (LUMINAIRE == i) {
            MAYBE_PRINT_CLIENT_ID(client_id);
            Serial.printf("c %d %lf\n", LUMINAIRE, node.get_cost());
          }
          else {
            MAYBE_ADD_CLIENT_ID(client_id, 0);
            enqueue_message(i, msg_t::GET_COST, data, MAYBE_ADD_CLIENT_SIZE(client_id, 0)); 
          }
          break;  

        // FIXME - commented because command 'c' for stage 2 was added
        // case 'c': /* Get controller parameters at desk <i> */
        //   if (LUMINAIRE == i) {
        //     Serial.printf("Controller parameters at luminaire %d:\n", LUMINAIRE);
        //     Serial.printf("K = %lf\n", controller.get_k());
        //     Serial.printf("Ti = %lf\n", controller.get_ti());
        //     Serial.printf("b = %lf\n", controller.get_b());
        //     Serial.printf("Tt = %lf\n", controller.get_tt());
        //   }
        //   break;

        default:
          MAYBE_PRINT_CLIENT_ID(client_id);
          Serial.println("err -> Invalid Command, please try again.");
          return;           
      }
    break;

    case 'r': /* Set illuminance reference at luminaire i */
      if (std::sscanf(buffer, "%c %d %lf", &cmd, &i, &val) == 3) {
        if (LUMINAIRE == i) {
          r = val;
          MAYBE_PRINT_CLIENT_ID(client_id);
          Serial.println("ack");
        }
        else {
          float aux = (float) val;
          memcpy(data, &aux, sizeof(aux));
          MAYBE_ADD_CLIENT_ID(client_id, sizeof(aux));
          enqueue_message(i, msg_t::SET_REFERENCE, data, MAYBE_ADD_CLIENT_SIZE(client_id, sizeof(aux)));
        }
      }
      else {
        master_calibrate_routine();
      }      
      break;

    case 'o': /* Set current occupancy state at desk <i> */
      std::sscanf(buffer, "%c %d %lf", &cmd, &i, &val);
      if (LUMINAIRE == i) {
        node.set_occupancy((int)val);
        MAYBE_PRINT_CLIENT_ID(client_id);
        Serial.println("ack");
        run_consensus();
      }
      else {
        float aux = (float) val;
        memcpy(data, &aux, sizeof(aux));
        MAYBE_ADD_CLIENT_ID(client_id, sizeof(aux));
        enqueue_message(i, msg_t::SET_OCCUPANCY, data, MAYBE_ADD_CLIENT_SIZE(client_id, sizeof(aux))); 
      }
      break;

    case 'a': /* Set anti-windup state at desk <i> */
      std::sscanf(buffer, "%c %d %lf", &cmd, &i, &val);
      if (LUMINAIRE == i) {
        controller.set_anti_windup(val);
        MAYBE_PRINT_CLIENT_ID(client_id);
        Serial.println("ack");
      }
      else {
        float aux = (float) val;
        memcpy(data, &aux, sizeof(aux));
        MAYBE_ADD_CLIENT_ID(client_id, sizeof(aux));
        enqueue_message(i, msg_t::SET_ANTI_WINDUP, data, MAYBE_ADD_CLIENT_SIZE(client_id, sizeof(aux))); 
      }
      break;

    case 'k': /* Set feedback on/off at desk <i> */
      std::sscanf(buffer, "%c %d %lf", &cmd, &i, &val);
      if (LUMINAIRE == i) {
        serial_duty_cycle = controller.get_u() / DAC_RANGE;
        controller.set_feedback(val);
        MAYBE_PRINT_CLIENT_ID(client_id);
        Serial.println("ack");
      }
      else {
        float aux = (float) val;
        memcpy(data, &aux, sizeof(aux));
        MAYBE_ADD_CLIENT_ID(client_id, sizeof(aux));
        enqueue_message(i, msg_t::SET_FEEDBACK, data, MAYBE_ADD_CLIENT_SIZE(client_id, sizeof(aux))); 
      }
      break;

    case 's': /* Start stream of real-time variable <x> of desk <i>. <x> can be “l”, “d” or "j". */
      std::sscanf(buffer, "%c %c %d", &cmd, &x, &i);
      Serial.printf("The current client id is %d\n", client_id);
      if (LUMINAIRE == i) {
        if (x == 'l')
          stream_l[LUMINAIRE].insert(client_id);
        else if (x == 'd')
          stream_d[LUMINAIRE].insert(client_id);
        else if (x == 'j')
          stream_j[LUMINAIRE].insert(client_id);
      }
      else {
        memcpy(data, &x, sizeof(x));
        MAYBE_ADD_CLIENT_ID(client_id, sizeof(x));
        enqueue_message(i, msg_t::START_STREAMING, data, MAYBE_ADD_CLIENT_SIZE(client_id, sizeof(x))); 
      }
      break;    
    
    case 'S': /* Stop stream of real-time variable <x> of desk <i>. <x> can be “l”, “d” or "j". */
      std::sscanf(buffer, "%c %c %d", &cmd, &x, &i);
      if (LUMINAIRE == i) {
        if (x == 'l') {
          if (!client_id) 
            stream_l[LUMINAIRE] = {};
          else
            set_remove(stream_l[LUMINAIRE], client_id);
        }
        else if (x == 'd') {
          if (!client_id) 
            stream_d[LUMINAIRE] = {};
          else
            set_remove(stream_d[LUMINAIRE], client_id);
        }
        else if (x == 'j') {
          if (!client_id) 
            stream_j[LUMINAIRE] = {};
          else
            set_remove(stream_j[LUMINAIRE], client_id);
        }
        MAYBE_PRINT_CLIENT_ID(client_id);
        Serial.println("ack");
      }
      else {
        memcpy(data, &x, sizeof(x));
        MAYBE_ADD_CLIENT_ID(client_id, sizeof(x));
        enqueue_message(i, msg_t::STOP_STREAMING, data, MAYBE_ADD_CLIENT_SIZE(client_id, sizeof(x))); 
      }
      break;  

    case 'v': /* Visualization with serial plotter */
      std::sscanf(buffer, "%c %d", &cmd, &i);
      if (LUMINAIRE == i) {
        visualization = !visualization;
        MAYBE_PRINT_CLIENT_ID(client_id);
        Serial.println("ack");
      }
      else {
        MAYBE_PRINT_CLIENT_ID(client_id);
        Serial.println("err");
      }
      break;
    
    case 'm': /* Set mode of operation */
      std::sscanf(buffer, "%c %c %d", &cmd, &subcmd, &i);
      if (LUMINAIRE == i)
        controller.set_modeOp(subcmd, i);
      MAYBE_PRINT_CLIENT_ID(client_id);
      Serial.println("ack");
      break;    

    case 'O': /* Set lower bound on illuminance for Occupied state at desk <i> */
      std::sscanf(buffer, "%c %d %lf", &cmd, &i, &val);
      if (LUMINAIRE == i) {
        node.set_lower_bound_occupied(val);
        MAYBE_PRINT_CLIENT_ID(client_id);
        Serial.println("ack");
        if (node.get_occupancy())
          run_consensus();
      }
      else {
        float aux = (float) val;
        memcpy(data, &aux, sizeof(aux));
        MAYBE_ADD_CLIENT_ID(client_id, sizeof(aux));
        enqueue_message(i, msg_t::SET_LOWER_BOUND_OCCUPIED, data, MAYBE_ADD_CLIENT_SIZE(client_id, sizeof(aux)));
      }
      break;  

    case 'U': /* Set lower bound on illuminance for Unoccupied state at desk <i> */
      std::sscanf(buffer, "%c %d %lf", &cmd, &i, &val);
      if (LUMINAIRE == i) {
        node.set_lower_bound_unoccupied(val);
        MAYBE_PRINT_CLIENT_ID(client_id);
        Serial.println("ack");
        if (!node.get_occupancy())
          run_consensus();
      }
      else {
        float aux = (float) val;
        memcpy(data, &aux, sizeof(aux));
        MAYBE_ADD_CLIENT_ID(client_id, sizeof(aux));
        enqueue_message(i, msg_t::SET_LOWER_BOUND_UNOCCUPIED, data, MAYBE_ADD_CLIENT_SIZE(client_id, sizeof(aux)));
      }
      break;

    case 'c': 
      if (sizeof(buffer) == 4) { /* Set current energy cost at desk <i> */
        std::sscanf(buffer, "%c %d %lf", &cmd, &i, &val);
        if (LUMINAIRE == i) {
          node.set_cost(val);
          MAYBE_PRINT_CLIENT_ID(client_id);
          Serial.println("ack");
          run_consensus();
        }
        else {
          float aux = (float) val;
          memcpy(data, &aux, sizeof(aux));
          MAYBE_ADD_CLIENT_ID(client_id, sizeof(aux));
          enqueue_message(i, msg_t::SET_COST, data, MAYBE_ADD_CLIENT_SIZE(client_id, sizeof(aux)));
        }
      }
      else if (sizeof(buffer) == 5) { /* Change controller parameters at desk <i> */
        std::sscanf(buffer, "%c %c %d %lf", &cmd, &subcmd, &i, &val);
        MAYBE_PRINT_CLIENT_ID(client_id);
        if (LUMINAIRE == i) {
          switch (subcmd) {
            case 'k': /* Change gain K at luminaire i controller */            
              controller.set_k(val);
              Serial.println("ack");
              break;

            case 'i': /* Change time constant Ti at luminaire i controller */
              controller.set_ti(val);
              Serial.println("ack");
              break;

            case 'b': /* Change gain b at luminaire i controller */
              controller.set_b(val);
              Serial.println("ack");
              break;

            case 't': /* Change time constant Tt at luminaire i controller */
              controller.set_tt(val);
              Serial.println("ack");
              break;
              
            default:
              Serial.println("err -> Invalid Command, please try again.");
              return; 
          }
        }
        else
          Serial.println("err -> Not implemented yet");
      }
      break;

    default:
      MAYBE_PRINT_CLIENT_ID(client_id);
      Serial.println("err -> Invalid Command, please try again.");
      return;      
  }
}

#undef MAYBE_PRINT_CLIENT_ID
#undef MAYBE_ADD_CLIENT_ID
#undef MAYBE_ADD_CLIENT_SIZE