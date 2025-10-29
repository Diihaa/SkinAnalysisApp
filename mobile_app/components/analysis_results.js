import { StatusBar } from 'expo-status-bar';
import React, { Component }  from 'react';
import { Linking, StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity, FlatList, ScrollView} from 'react-native';
import { Octicons, FontAwesome, AntDesign, Ionicons, Feather, Entypo } from '@expo/vector-icons';
import axios from 'axios';
// import {Backend_Url} from './backend_url'
// secure store import
import * as SecureStore from 'expo-secure-store';

class AnalysisResults extends Component {
    constructor(props) {
        super(props);
        this.state = {
            user_access_token: '',
            skin_type_match: '',
            skin_disorder_match: '',
            user_allergic_to: '',
            done: false,
            recommended_products_list: [],
            recommended_routine: {}
        }

        this.HandleChange = (value, state) => {
            this.setState({ [state]: value })
        }

        this.GetBackendURL = async () => {
            var should_reload = false

            // get backend url
            let backend_url = await SecureStore.getItemAsync('backend_url');
            if (backend_url){
              this.setState({backend_url: backend_url})
            }else{ should_reload = true }

            if (should_reload == true){  const timeoutId = setTimeout(() => {this.GetBackendURL()}, 1000) }
        }

        this.GetUserDetails = async () => {
            var should_reload = false

            // get access token
            let token = await SecureStore.getItemAsync('token');
            if (token){
              this.setState({user_access_token: token})
            }else{ should_reload = true }

            // get username
            let user_name = await SecureStore.getItemAsync('user_name');
            if (user_name){
              this.setState({user_name: user_name})
            }else{ should_reload = true }

            if (should_reload == true){  const timeoutId = setTimeout(() => {this.GetUserDetails()}, 1000) }
        }

        this.GetAnalysisResults = () => {
            var { skin_type_match, skin_disorder_match, recommended_products_list, recommended_routine, user_allergic_to, done } = this.props.route.params;
    
            if (done == true){
    
                // skin_type_match = JSON.stringify(skin_type_match)
                // skin_disorder_match = JSON.stringify(skin_disorder_match) 
                
                this.setState({
                    skin_type_match: skin_type_match,
                    skin_disorder_match: skin_disorder_match,
                    recommended_products_list: recommended_products_list,
                    recommended_routine: recommended_routine,
                    user_allergic_to: user_allergic_to,
                    done: done
                })
    
            }else{
    
                this.setState({done: done})
                
            }
        }
    };

    componentDidMount() {
        this.GetAnalysisResults()
        // get backend url
        this.GetBackendURL()
        // get user details
        this.GetUserDetails()
    }


    render() {
        var Backend_Url = this.state.backend_url
        var products_map = this.state.recommended_products_list.map((item, index) => {
            return<View key={'prod'+index}
                onStartShouldSetResponder={() => this.props.navigation.navigate('Product', {product: item})}
                style={{margin: 4, marginBottom: 20, height: 100, flexDirection: 'row', borderBottomColor: '#eaeaea', borderBottomWidth: 1}}
            >
                <View style={{height: 65, flexDirection: 'column', backgroundColor: '#000000', width: 90}}>
                    <Image source={{uri: Backend_Url + 'media/' + item.image,}} style={{width: 90, height: 90}} />
                </View>
                <View style={{flexDirection: 'column', height: 65, borderBottomLeftRadius: 5, borderBottomRightRadius: 5, width: '70%'}}>
                    <View style={{margin: 10}}>
                        {
                            item.name.length <= 23
                            ? <Text style={{textAlign: 'left', fontSize: 15}}>
                                {item.name}
                            </Text>
                            : <Text style={{textAlign: 'left', fontSize: 15}}>
                                {item.name.substring(0, 21)}...
                            </Text>
                        }
                        {
                            item.target_skin_types != ''
                            ? <View>
                                {
                                    item.target_skin_types.length <= 16
                                    ? <Text style={{textAlign: 'left', marginTop: 5, marginLeft: 0, color: '#3f4144', fontSize: 12}}>
                                        Target skin type(s): {item.target_skin_types}
                                    </Text>
                                    : <Text style={{textAlign: 'left', marginTop: 5, marginLeft: 0, color: '#3f4144', fontSize: 12}}>
                                        Target skin type(s): {item.target_skin_types.substring(0, 13)}...
                                    </Text>
                                }
                            </View>
                            : <View>

                            </View>
                        }
                        {
                            item.target_skin_disorders != ''
                            ? <View>
                                {
                                    item.target_skin_disorders.length <= 16
                                    ? <Text style={{textAlign: 'left', marginTop: 5, marginLeft: 0, color: '#3f4144', fontSize: 12}}>
                                        Target skin disorder(s): {item.target_skin_disorders}
                                    </Text>
                                    : <Text style={{textAlign: 'left', marginTop: 5, marginLeft: 0, color: '#3f4144', fontSize: 12}}>
                                        Target skin disorder(s): {item.target_skin_disorders.substring(0, 13)}...
                                    </Text>
                                }
                            </View>
                            : <View>

                            </View>
                        }
                        <Text style={{alignSelf: "flex-start", position: "absolute", right: 5, top: 8}}>
                            ${item.price}
                        </Text>
                    </View>
                </View>
            </View>
        })

        var recommended_routine = this.state.recommended_routine

        return<View style={styles.container}>
            <View style={styles.top_bar}>
                <View style={{marginLeft: 10, marginRight: 10, marginTop: 20}}>
                    <Text style={{fontSize: 13, color: '#3f4144', textAlign: 'left'}}>
                        Skin type match: {this.state.skin_type_match}
                    </Text>
                    <Text style={{fontSize: 13, color: '#3f4144', textAlign: 'left', marginTop: 10}}>
                        Skin disorder match: {this.state.skin_disorder_match}
                    </Text>
                    <Text style={{fontSize: 13, color: '#3f4144', textAlign: 'left', marginTop: 10}}>
                        Your known allergies: {this.state.user_allergic_to}
                    </Text>
                </View>
            </View>
            <ScrollView style={styles.scroll_view}>
                <View style={{marginTop: 20, marginBottom: 20}}>
                    <Text style={{fontSize: 15, color: '#3f4144', textAlign: 'center', fontWeight: 'bold'}}>
                        Recommended skincare routine: 
                    </Text>
                </View>
                <View style={{marginTop: 10}}>
                    <Text style={{fontSize: 15, color: '#3f4144', textAlign: 'left', fontWeight: 'bold'}}>
                        Cleansing: 
                    </Text>
                    <Text style={{fontSize: 13, color: '#3f4144', textAlign: 'left', marginTop: 5}}>
                        {recommended_routine['Cleansing']}
                    </Text>
                </View>
                <View style={{marginTop: 10}}>
                    <Text style={{fontSize: 15, color: '#3f4144', textAlign: 'left', fontWeight: 'bold'}}>
                        Exfoliation: 
                    </Text>
                    <Text style={{fontSize: 13, color: '#3f4144', textAlign: 'left', marginTop: 5}}>
                        {recommended_routine['Exfoliation']}
                    </Text>
                </View>
                <View style={{marginTop: 10}}>
                    <Text style={{fontSize: 15, color: '#3f4144', textAlign: 'left', fontWeight: 'bold'}}>
                        Moisturizing: 
                    </Text>
                    <Text style={{fontSize: 13, color: '#3f4144', textAlign: 'left', marginTop: 5}}>
                        {recommended_routine['Moisturizing']}
                    </Text>
                </View>
                <View style={{marginTop: 10}}>
                    <Text style={{fontSize: 15, color: '#3f4144', textAlign: 'left', fontWeight: 'bold'}}>
                        Lifestyle: 
                    </Text>
                    <Text style={{fontSize: 13, color: '#3f4144', textAlign: 'left', marginTop: 5}}>
                        {recommended_routine['Lifestyle']}
                    </Text>
                </View>
                <View style={{marginTop: 30, marginBottom: 20}}>
                    <Text style={{fontSize: 15, color: '#3f4144', textAlign: 'center', fontWeight: 'bold'}}>
                        Recommended Products: 
                    </Text>
                </View>
                {
                    this.state.recommended_products_list.length == 0
                    ? <View>
                        {
                            this.state.done == false
                            ? <Text style={{color: '#00539C', fontWeight: 'bold', marginTop: 80, textAlign: 'center', marginBottom: 100}}>
                                You have to submit an image for analysis first inorder to view results
                            </Text>
                            : <Text style={{color: '#00539C', fontWeight: 'bold', marginTop: 80, textAlign: 'center', marginBottom: 100}}>
                                No product recommendations found
                            </Text>
                        }
                    </View>
                    : <View>
                        {products_map}
                    </View>
                }
            </ScrollView>
        </View>
    }

}

export default AnalysisResults;

const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#ffffff',
    //   textAlign: 'left',
    //   justifyContent: 'center'
    },
    search_bar: {
        // borderBottomWidth: 0.1,
        // borderBottomColor: '#eaeaea',
        flexDirection: 'row',
        backgroundColor: '#f2f2f2',
        // alignItems: 'center',
        // justifyContent: 'center',
        height: 50,
        marginTop: 50,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        top: 0
    },
    top_bar: {
        borderBottomWidth: 0.1,
        borderBottomColor: '#eaeaea',
        // flexDirection: 'row',
        backgroundColor: '#ffffff',
        // alignItems: 'center',
        // justifyContent: 'center',
        height: 80,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        top: 0,
    },
    scroll_view: {
        height: 2,
        marginTop: 90,
        marginLeft: 10,
        marginRight: 10
    },
    bottom_bar: {
        flexDirection: 'row',
        backgroundColor: '#ffffff',
        alignItems: 'center',
        justifyContent: 'center',
        height: 70,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        bottom: 0
    }
});