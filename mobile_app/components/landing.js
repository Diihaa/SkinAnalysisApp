import { StatusBar } from 'expo-status-bar';
import React, { Component }  from 'react';
import { Linking, StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity, FlatList, ScrollView} from 'react-native';
import { Octicons, FontAwesome, AntDesign, Ionicons, Feather, Entypo } from '@expo/vector-icons';
import axios from 'axios';
import {Backend_Url} from './backend_url'
import Logo from '../assets/icon.png'

class Landing extends Component {
    constructor(props) {
        super(props);
        this.state = {
            
        }
    };

    componentDidMount() {

    }


    render() {

        return<View style={styles.container}>
            <Image source={Logo} style={{width: 150, height: 150, marginLeft: 'auto', marginRight: 'auto', marginTop: 150, borderRadius: 20}}/>
            <TouchableOpacity
                key='login'
                onPress={() => this.props.navigation.navigate('Signin')}
                style={{backgroundColor: '#ffffff', marginLeft: 'auto', marginRight: 'auto', marginTop: 60, marginBottom: 0, width: '90%', height: 50, borderRadius: 5, borderWidth: 1, borderColor: 'grey'}}
            >
                <Text style={{fontWeight: 'bold', color: '#3f4144', fontSize: 17, marginLeft: 'auto', marginRight: 'auto', marginBottom: 'auto', marginTop: 'auto', textAlign: 'center'}}>
                    Login
                </Text>
            </TouchableOpacity>
            <TouchableOpacity
                key='signup'
                onPress={() => this.props.navigation.navigate('Signup')}
                style={{backgroundColor: '#00539C', marginLeft: 'auto', marginRight: 'auto', marginTop: 20, marginBottom: 0, width: '90%', height: 50, borderRadius: 5}}
            >
                <Text style={{fontWeight: 'bold', color: '#ffffff', fontSize: 17, marginLeft: 'auto', marginRight: 'auto', marginBottom: 'auto', marginTop: 'auto', textAlign: 'center'}}>
                    Create Account
                </Text>
            </TouchableOpacity>
            {/* <View style={styles.bottom_bar}>
                <Text style={{color: '#00539C', fontSize: 15, fontWeight: 'bold', marginLeft: 'auto', marginRight: 'auto', marginTop: 5, textAlign: 'center'}}>
                    By
                </Text>
                <Text style={{color: '#00539C', fontSize: 15, fontWeight: 'bold', marginLeft: 'auto', marginRight: 'auto', marginTop: 5, textAlign: 'center'}}>
                    Michael
                </Text>
            </View> */}
        </View>
    }

}

export default Landing;

const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#ffffff',
    //   textAlign: 'left',
    //   justifyContent: 'center'
    },
    scroll_view: {
        height: 2
    },
    bottom_bar: {
        // borderTopWidth: 1,
        // borderTopColor: 'gold',
        // flexDirection: 'row',
        backgroundColor: '#ffffff',
        alignItems: 'center',
        justifyContent: 'center',
        height: 60,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        bottom: 0
    }
});